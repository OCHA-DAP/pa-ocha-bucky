import argparse
import csv
import datetime
import glob
import logging
import os
import pickle
import sys
from datetime import timedelta

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import tqdm

from .util import estimate_IFR, TqdmLoggingHandler
from .util.read_config import bucky_cfg
from .util.update_data_repos import update_repos

# TODO all these paths should be combined properly rather than just with str cat

DAYS_OF_HIST = 45

# Initialize argument parser
parser = argparse.ArgumentParser(description="Bucky Model input graph creation")

# Required: Start date of simulation
parser.add_argument(
    "-d",
    "--date",
    default=str(datetime.date.today() - timedelta(days=14)),
    type=str,
    help="Start date of simulation, last date for historical data.",
)

parser.add_argument(
    "-o",
    "--output",
    default=bucky_cfg["data_dir"] + "/input_graphs/",
    type=str,
    help="Directory for graph file. Defaults to data/input_graphs/",
)

parser.add_argument(
    "--hist_file",
    default=bucky_cfg["data_dir"] + "/cases/csse_hist_timeseries.csv",
    type=str,
    help="File to use for historical data",
)

parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    dest="verbosity",
    default=0,
    help="verbose output (repeat for increased verbosity; defaults to WARN, -v is INFO, -vv is DEBUG)",
)

parser.add_argument("--no_update", action="store_false", help="Skip updating data")


def get_case_history(historical_data, end_date, num_days=DAYS_OF_HIST):
    """Gets case and death history for the requested number of days for 
    each FIPS.

    If data is missing for a date, it is replaced with the data from the 
    last valid date.
    
    Parameters
    ----------
    historical_data : Pandas DataFrame
        Dataframe with case, death data indexed by date, FIPS
    end_date : date string
        Last date to get data for
    num_days : int
        Number of days of history requested
    
    Returns
    -------
    hist : dict
        Dictionary of case data, keyed by FIPS
    """

    # Cast historical data's dates to datetime objects
    historical_data["date"] = pd.to_datetime(historical_data["date"])
    end_date = pd.to_datetime(end_date)
    start_date = end_date - timedelta(days=num_days)
    hist = {}

    logging.info(
        "Getting " + str(num_days) + " days of case/death data for each county..."
    )

    for fips, group in tqdm.tqdm(historical_data.groupby("adm2"), desc="Grabbing adm2 histories", dynamic_ncols=True):

        # Get block of data
        block = group.loc[(group["date"] >= start_date) & (group["date"] <= end_date)]
        block = block[["cumulative_reported_cases", "cumulative_deaths", "date"]]

        # If no data, fill with zeros
        if block.empty:
            hist[fips] = np.zeros(num_days + 1)
        else:
            # Sort by data
            block = block.set_index("date").sort_index()

            # Get array of values
            confirmed = block["cumulative_reported_cases"].to_numpy()
            deaths = block["cumulative_deaths"].to_numpy()

            # TODO this needs a refactor
            # If there are less than the requested values, fill in missing values
            if len(confirmed) < (num_days + 1):

                # Create nan frame to fill missing values with nans
                nan_frame = pd.DataFrame(
                    index=pd.date_range(start=start_date, end=end_date, freq="1d")
                )

                # Merge with slice
                block = nan_frame.merge(
                    block, left_index=True, right_index=True, how="left"
                ).bfill()
                confirmed = block["cumulative_reported_cases"].to_numpy()
            if len(deaths) < (num_days + 1):

                # Create nan frame to fill missing values with nans
                nan_frame = pd.DataFrame(
                    index=pd.date_range(start=start_date, end=end_date, freq="1d")
                )

                # Merge with slice
                block = nan_frame.merge(
                    block, left_index=True, right_index=True, how="left"
                ).bfill()
                deaths = block["cumulative_deaths"].to_numpy()

            hist[fips] = np.vstack([confirmed, deaths])

    return hist


def compute_population_density(age_df, shape_df):
    """Computes normalized population density.
    
    Parameters
    ----------
    age_df : Pandas DataFrame
        age-stratified population data
    shape_df : Geopandas GeoDataFrame
        GeoDataFrame with shape information indexed by FIPS
    
    Returns
    -------
    popdens : Pandas DataFrame
        DataFrame with population density by FIPS

    """
    # Use age data and shape file to compute population density
    pop_df = pd.DataFrame(age_df.sum(axis=1))
    pop_df.rename(columns={0: "total"}, inplace=True)
    popdens = pop_df.merge(
        counties.set_index("adm2")["ALAND"].to_frame(),
        left_index=True,
        right_index=True,
    )
    popdens = (popdens["total"] / popdens["ALAND"]).to_frame(name="pop_dens")

    # Scale so values are relative
    popdens["pop_dens_scaled"] = popdens["pop_dens"] / popdens["pop_dens"].max()

    return popdens


def read_descartes_data(end_date):
    """Reads Descartes mobility data. :cite:`warren2020mobility`
    
    Parameters
    ----------
    end_date : string
        Last date to get Descartes data

    Returns
    -------
    nat_frac_move : Pandas DataFrame
        TODO
    dl_state : Pandas DataFrame
        TODO
    dl_county : Pandas DataFrame
        TODO

    Notes
    -----
    Data provided by Descartes Labs (https://descarteslabs.com/mobility/) [1]_

    .. [1] Warren, Michael S. & Skillman, Samuel W. "Mobility Changes in Response to COVID-19". arXiv:2003.14228 [cs.SI], Mar. 2020. arxiv.org/abs/2003.14228
    """
    dl_data = pd.read_csv(mobility_dir + "DL-us-m50_index.csv")
    dl_data.rename(columns={"fips": "adm2"}, inplace=True)
    dl_data = (
        dl_data.set_index(["admin_level", "adm2"])
        .drop(columns=["country_code", "admin1", "admin2"])
        .stack()
        .reset_index()
        .rename(columns={"level_2": "date"})
    )

    # Get last 7 days worth of data
    last7days = np.sort(dl_data["date"].unique())
    last7days = last7days[last7days < end_date][-7:]
    dl_data = dl_data[dl_data.date.isin(last7days)]

    # Get national-level data
    dl_data_nat = dl_data[dl_data.admin_level == 0]

    # Compute national move fraction (net fraction of people that moved)
    nat_frac_move = dl_data_nat.mean()[0] / 100.0

    # Compute for fips
    dl_data = dl_data.groupby("adm2").mean().rename(columns={0: "frac_move"})
    dl_data["frac_move"] = dl_data["frac_move"] / 100.0

    # Drop national level and cast index to int
    dl_data = dl_data[dl_data.admin_level > 0]  # drop national level
    dl_data.index = dl_data.index.astype(int)

    dl_data["frac_move"] = dl_data["frac_move"].clip(lower=0.2)

    # Get state and county level
    dl_state = dl_data[dl_data.admin_level == 1]
    dl_county = dl_data[dl_data.admin_level == 2]

    return nat_frac_move, dl_state, dl_county


def read_lex_data(date):
    """Reads county-level location exposure indices for a given date from
    PlaceIQ location data.

    In order to improve performance, preprocessed data is saved. If the 
    user requests data for a date that has already been preprocessed, it
    will read the data from disk instead of repeating the processing.
    
    Parameters
    ----------
    date : string
        Fetches data for requested date
    
    Returns
    -------
    df_long : Pandas DataFrame
        Preprocessed LEX data

    """
    # Check if a preprocessed version already exists
    cache_file = bucky_cfg["data_dir"] + "/mobility/preprocessed/county_lex_" + date + ".csv.gz"
    if os.path.exists(cache_file):
        df_long = pd.read_csv(cache_file)
        logging.info("Using cached lex data for " + str(date))
    else:
        df = pd.read_csv(
            bucky_cfg["data_dir"] + "/mobility/COVIDExposureIndices/lex_data/county_lex_"
            + date
            + ".csv.gz",
            compression="gzip",
            header=0,
        )

        logging.info(str(date) + " not in cache")
        counties = df.columns.values[1:]
        col_names = dict(zip(counties, ["a" + lab for lab in counties]))
        df = df.rename(columns=col_names)
        df_long = pd.wide_to_long(df, stubnames="a", i=["COUNTY_PRE"], j="col")
        df_long = df_long.reset_index(drop=False)
        df_long = df_long.rename(
            columns={"COUNTY_PRE": "StartId", "col": "EndId", "a": "frac_count"}
        )

        # Save processed file
        if not os.path.exists(bucky_cfg["data_dir"] + "/mobility/preprocessed"):
            os.makedirs(bucky_cfg["data_dir"] + "/mobility/preprocessed")

        df_long.to_csv(bucky_cfg["data_dir"] + "/mobility/preprocessed/county_lex_" + date + ".csv.gz")

    return df_long


def get_lex(last_date, window_size=7):
    """Reads county-level location exposure indices from PlaceIQ 
    location data and applies a window.
    
    Parameters
    ----------
    last_date : last_date
        Fetches data for requested date
    window_size : int (default: 7)
        Size of window, in days, to apply to data
    Returns
    -------
    frac_df : Pandas DataFrame
        TODO

    """
    lex_df = None
    success = 0
    d = 0
    pbar = tqdm.tqdm(total=window_size, desc='Getting historical LEX', dynamic_ncols=True)
    while success < window_size:
        date = datetime.date.fromisoformat(last_date) - datetime.timedelta(days=d)
        date_str = date.isoformat()
        logging.info(date_str)

        d += 1
        try:
            if lex_df is None:
                lex_df = read_lex_data(date_str)
                lex_df.set_index(["StartId", "EndId"], inplace=True)
                lex_df.rename(columns={"frac_count": date_str}, inplace=True)
            else:
                tmp_df = read_lex_data(date_str)
                tmp_df.set_index(["StartId", "EndId"], inplace=True)
                tmp_df.rename(columns={"frac_count": date_str}, inplace=True)
                lex_df = lex_df.merge(
                    tmp_df, left_index=True, right_index=True, how="outer"
                )
            success += 1
            pbar.update(1)
        except FileNotFoundError as e:
            # print(e)
            continue

    lex_df.fillna(0.0, inplace=True)
    mean_df = lex_df.mean(axis=1)
    tot_df = mean_df.reset_index().groupby("StartId").sum()

    frac_df = mean_df.divide(tot_df[0], axis=0, level=0)
    return frac_df.to_frame(name="frac_count").reset_index()


def get_safegraph(last_date, window_size=7):
    """Reads SafeGraph mobility data and applies a window.
    
    Parameters
    ----------
    last_date : last_date
        Fetches data for requested date
    window_size : int (default: 7)
        Size of window, in days, to apply to data
    Returns
    -------
    frac_df : Pandas DataFrame
        TODO

    """
    sg_df = None
    success = 0
    d = 0
    pbar = tqdm.tqdm(total=window_size, desc='Getting historical SG', dynamic_ncols=True)
    while success < window_size:

        date = datetime.date.fromisoformat(last_date) - datetime.timedelta(days=d)
        date_str = date.isoformat()
        d += 1
        try:
            if sg_df is None:
                sg_df = pd.read_csv(
                    bucky_cfg["data_dir"] + "/safegraph_processed/" + date_str + "_county.csv.gz"
                )
                sg_df.set_index(["origin", "dest"], inplace=True)
                sg_df.rename(columns={"count": date_str}, inplace=True)
            else:
                tmp_df = pd.read_csv(
                    bucky_cfg["data_dir"] + "/safegraph_processed/" + date_str + "_county.csv.gz"
                )
                tmp_df.set_index(["origin", "dest"], inplace=True)
                tmp_df.rename(columns={"count": date_str}, inplace=True)
                sg_df = sg_df.merge(
                    tmp_df, left_index=True, right_index=True, how="outer"
                )
            logging.info("using sg data from " + date_str)
            success += 1
            pbar.update(1)
        except FileNotFoundError:
            continue

    sg_df.fillna(0.0, inplace=True)
    mean_df = sg_df.mean(axis=1)
    tot_df = mean_df.reset_index().groupby("origin").sum()

    frac_df = mean_df.divide(tot_df[0], axis=0, level=0)
    return frac_df


def get_mobility_data(popdens, end_date, age_data, add_territories=True):
    """Fetches mobility data.
    
    Parameters
    ----------
    popdens : Pandas DataFrame
        Population density indexed by FIPS
    end_date : string
        Last date of historical data
    age_data : Pandas DataFrame
        County-level age-stratified population data
    add_territories : boolean
        Adds territory data if True

    Returns
    -------
    mean_edge_weights : Pandas DataFrame
        TODO
    move_dict : dict
        TODO

    """
    # lex = read_lex_data(last_date)
    lex = get_lex(last_date)

    national_frac_move, dl_state, dl_county = read_descartes_data(last_date)

    if os.path.exists(os.path.join(bucky_cfg["data_dir"], "safegraph_processed")):

        sg_df = get_safegraph(last_date)
        tmp = lex.set_index(["StartId", "EndId"]).frac_count.sort_index()
        sg_df.index.rename(["StartId", "EndId"], inplace=True)
        merged_df = tmp.to_frame().merge(
            sg_df.to_frame(), left_index=True, right_index=True, how="outer"
        )
        mean_df = merged_df.mean(axis=1, skipna=True)
        lex = mean_df.to_frame(name="frac_count").reset_index()

    # Combine Teralytics, Descartes data
    state_map = counties[["adm2", "adm1"]].set_index("adm2")
    lex = lex.reset_index().set_index("StartId")
    lex = lex.merge(state_map, left_index=True, right_index=True, how="left")

    # Merge state movement
    lex = lex.merge(dl_state, left_on="adm1", right_index=True, how="left")

    # Add in the counties that we have
    # TODO Replace with loc for performance
    lex.update(dl_county)

    # use national to cover any nans (like PR)
    lex[lex.frac_move.isna()] = lex[lex.frac_move.isna()].assign(
        frac_move=national_frac_move
    )

    # Fix index
    lex = lex.reset_index().rename(
        columns={"level_0": "StartId"}
    )  # .set_index(['StartId', 'EndId'])

    lex = lex.merge(popdens, left_on="EndId", right_index=True, how="left")
    lex["frac_count"] = lex["frac_count"] * np.sqrt(
        np.maximum(0.02, lex["pop_dens_scaled"]) ** 2
    )

    # Use data to make mean edge weights
    mean_edge_weights = lex.groupby(["StartId", "EndId"]).mean().dropna()

    # Compare the difference between before & after
    movement = lex.set_index(["StartId", "EndId"])["frac_move"]
    # TODO this takes forever:
    move_dict = {ind: {"R0_frac": movement.loc[ind]} for ind in movement.index}

    return mean_edge_weights, move_dict


if __name__ == "__main__":

    # Parse cli args
    args = parser.parse_args()

    loglevel = 30 - 10 * min(args.verbosity, 2)

    # Logging
    logging.basicConfig(
        level=loglevel,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
        handlers=[TqdmLoggingHandler()],
    )

    logging.info(args)

    # Define some parameters
    last_date = args.date
    start = datetime.datetime.now()
    pickle_id = str(start).replace(" ", "__").replace(":", "_").split(".")[0]
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "usa--" + pickle_id + ".p")
    update_data = args.no_update

    # Data Files
    county_shapefile = bucky_cfg["data_dir"] + "/shapefiles/tl_2019_us_county.shp"
    age_strat_file = bucky_cfg["data_dir"] + "/population/US_pop.csv"
    mobility_dir = bucky_cfg["data_dir"] + "/mobility/DL-COVID-19/"
    contact_mat_folder = (
        bucky_cfg["data_dir"] + "/contact_matrices_152_countries/*2.xlsx"
    )
    state_mapping_file = bucky_cfg["data_dir"] + "/statefips_to_names.csv"

    ##### FILE MANAGEMENT #####
    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    ##### UPDATE DATA #####
    # Update remote data repos for case data and mobility data
    if update_data:
        update_repos()

    ##### START GRAPH CREATION #####
    ##### SHAPE FILE #####
    # Read county data
    counties = gpd.read_file(county_shapefile)
    counties["adm2"] = counties["GEOID"].astype(int)
    counties["adm1"] = counties["STATEFP"].astype(int)

    ##### GEOID MAPPING #####

    # Create dict for statefips to name
    state_map = csv.DictReader(open(state_mapping_file))
    statefp_to_name = {}

    for row in state_map:
        statefp_to_name[int(row["statefips"])] = row["name"]

    ##### AGE AND DEMO DATA #####
    # Read age-stratified data
    age_data = pd.read_csv(age_strat_file, index_col=0, header=[0, 1])
    age_data.index = age_data.index.astype(int)

    # Add age-stratified data for territories
    territory_df = pd.read_csv(
        bucky_cfg["data_dir"] + "/population/territory_pop.csv", index_col="fips"
    )
    territory_df.index = territory_df.index.astype(int)

    # TODO Temp use the mean age to get IFR until we can regenerate territory_pop.csv
    if not isinstance(territory_df.columns, pd.MultiIndex):
        terr_ifr_ind = territory_df.index
        mean_bin_age = (np.arange(0, 80, 5) + np.arange(5, 85, 5)) / 2
        mean_bin_age[-1] = 82  # just an estimate...
        ifr_series = pd.Series(estimate_IFR(mean_bin_age), index=territory_df.columns)
        ifr_df = pd.DataFrame({fips: ifr_series for fips in territory_df.index}).T

        territory_df = pd.concat([territory_df, ifr_df], axis=1, keys=["N", "IFR"])

    age_data = age_data.append(territory_df)

    # Create a dict (for node attribute)
    # TODO this is be a to_dict or something rather than this loop and loc[]...
    age_dict = {
        fips: {
            "N_age_init": age_data.N.loc[fips].values,
            "Population": np.sum(age_data.N.loc[fips].values),
            "IFR": age_data.IFR.loc[fips].values,
        }
        for fips in age_data.index
    }

    # Compute population density
    popdens = compute_population_density(age_data, counties)

    ##### CASE DATA #####
    # Read historical data
    hist_data = pd.read_csv(args.hist_file)

    # If last date was not provided, use last date in historical data
    if last_date is None:
        last_date = hist_data.date.max()

    # Get historical case data for each county
    hist_per_fips = get_case_history(hist_data, last_date)

    # Check that historical data is always increasing - print warning if not
    non_increasing_fips = []
    for f in hist_per_fips:

        confirmed_arr = hist_per_fips[f][0]
        deaths_arr = hist_per_fips[f][1]

        # Check if always increasing
        if not np.all(np.diff(confirmed_arr) >= 0) or not np.all(
            np.diff(confirmed_arr) >= 0
        ):
            non_increasing_fips.append(f)

    if len(non_increasing_fips) > 0:

        logging.warning("Some adm2 have non-monotonically increasing historical data.")

    # Get historical data for start date of the simulation
    date_data = hist_data.set_index(["adm2", "date"]).xs(last_date, level=1)

    # grab from covid tracking project, (only defined at state level)
    ct_data = pd.read_csv(bucky_cfg["data_dir"] + "/cases/covid_tracking.csv")
    ct_data = ct_data.loc[ct_data.date <= last_date]
    ct_data.set_index(["adm1", "date"], inplace=True)

    # Remove duplicates
    # TODO: Find cause of duplicates
    date_data = date_data.loc[~date_data.index.duplicated(keep="first")]

    ##### MOBILITY DATA #####

    # Get mobility data
    mean_edge_weights, move_dict = get_mobility_data(popdens, last_date, age_data)

    # Create list of edges
    edge_list = mean_edge_weights.reset_index()[["StartId", "EndId", "frac_count"]]

    ##### COMBINE DATA #####
    # Start merging all of our data together
    data = date_data.merge(popdens, left_on="adm2", right_index=True, how="left")

    # Add shape data
    # Check if there are counties present in data but missing shapes
    data_fips = data.index.values
    shape_fips = counties["adm2"].unique()
    fips_diff = np.setdiff1d(data_fips, shape_fips)
    if len(fips_diff) > 0:
        logging.warning(
            str(len(fips_diff))
            + " FIPS appear in historical data but do not have shapefile information. This data is dropped"
        )

    data = counties.merge(data, on="adm2", sort=True, how="left")

    for col in ["cumulative_reported_cases", "cumulative_deaths"]:
        data[col] = data[col].fillna(0.0)

    data = data[
        ["adm2", "adm1", "cumulative_reported_cases", "cumulative_deaths", "pop_dens", "geometry", "NAMELSAD"]
    ]
    data = data.rename(columns={"NAMELSAD": "adm2_name"})

    # Drop duplicates
    data.drop(data[data.duplicated("adm2", keep="first")].index, inplace=True)

    data.dropna(
        subset=["geometry"], inplace=True
    )  # this should happen but 2 of the cases have unknown fips

    # Only keep FIPS that appear in our age data
    data = data[data["adm2"].isin(list(age_dict.keys()))]

    # Ensure there are 3233 values
    required_num_nodes = 3233
    if data.shape[0] != required_num_nodes:

        logging.warning(
            str(required_num_nodes)
            + " nodes are expected. Only found "
            + str(data.shape[0])
        )

    ##### EDGE CREATION #####
    # Get list of all unique FIPS
    uniq_fips = pd.unique(data["adm2"])

    # Keep edges that have starts/ends in the list of FIPS
    edge_list = edge_list[edge_list["StartId"].isin(uniq_fips)]
    edge_list = edge_list[edge_list["EndId"].isin(uniq_fips)]
    edge_list = edge_list.to_numpy()

    # Create edges
    logging.info("Creating edges...")
    edges = []
    for index, row in tqdm.tqdm(data.iterrows(), total=len(data), desc='Creating neighboring edges', dynamic_ncols=True):

        # Determine which counties touch
        neighbors = data[data.geometry.touches(row["geometry"])].adm2.to_numpy()
        edges.append(
            np.vstack(
                [
                    np.full(neighbors.shape, row.adm2),
                    neighbors,
                    np.full(neighbors.shape, 0.001),
                ]
            )
        )  # *popdens['pop_dens'].loc[neighbors])]))

    # Drop geometry column
    data.drop(columns=["geometry"], inplace=True)

    diff_edge_list = np.hstack(edges).T

    self_loops = np.vstack(2 * [uniq_fips] + [np.ones(uniq_fips.shape)]).T
    edge_list = np.vstack([edge_list, diff_edge_list, self_loops])

    ##### CONTACT MATRICES #####
    # Initialize contact matrices
    contact_mats = {}

    # iterate over files and read
    for f in glob.glob(contact_mat_folder):
        mat = pd.read_excel(
            f, sheet_name="United States of America", header=None
        ).to_numpy()
        mat_name = "_".join(f.split("/")[-1].split("_")[1:-1])
        contact_mats[mat_name] = mat

    ##### FINAL GRAPH #####
    # Create graph
    G = nx.MultiDiGraph(contact_mats=contact_mats)
    G.add_weighted_edges_from(edge_list)

    node_attr_dict = data.set_index("adm2").to_dict("index")
    nx.set_node_attributes(G, node_attr_dict)
    nx.set_node_attributes(G, age_dict)

    hist_per_fips = {
        k: {"case_hist": v[0], "death_hist": v[1]}
        for k, v in hist_per_fips.items()
        if k in uniq_fips
    }
    for fips in uniq_fips:
        if fips in hist_per_fips:
            if "case_hist" not in hist_per_fips[fips]:
                hist_per_fips[fips]["case_hist"] = np.zeros((DAYS_OF_HIST + 1,))
            if "death_hist" not in hist_per_fips[fips]:
                hist_per_fips[fips]["death_hist"] = np.zeros((DAYS_OF_HIST + 1,))
        else:
            hist_per_fips[fips] = {
                "case_hist": np.zeros((DAYS_OF_HIST + 1,)),
                "death_hist": np.zeros((DAYS_OF_HIST + 1,)),
            }

    nx.set_node_attributes(G, hist_per_fips)

    # coalesce edges by summing weights
    G2 = nx.DiGraph(
        contact_mats=contact_mats,
        adm1_key="adm1",
        adm2_key="adm2",
        adm1_to_str=statefp_to_name,
        adm0_name="US",
        start_date=last_date,
        covid_tracking_data=ct_data,
    )
    G2.add_edges_from(G.edges(), weight=0.0, R0_frac=1.0)
    G2.update(nodes=G.nodes(data=True))

    logging.info("Finalizing edge weights...")
    for u, v, d in tqdm.tqdm(G.edges(data=True), total=len(G.edges), desc='Finalizing edges', dynamic_ncols=True):

        G2[u][v]["weight"] += d["weight"]
        if (u, v) in move_dict:
            G2[u][v]["R0_frac"] = move_dict[(u, v)]["R0_frac"]

    G_out = nx.convert_node_labels_to_integers(G2, label_attribute="adm2")

    logging.info("Writing output pickle: " + out_file)
    # Write to pickle
    with open(out_file, "wb") as f:
        pickle.dump(G_out, f)

    logging.info("Done!")
