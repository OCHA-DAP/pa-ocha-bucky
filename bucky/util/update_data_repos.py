import glob
import logging
import os
import ssl
import subprocess
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import tqdm

from .read_config import bucky_cfg

# Options for correcting territory data
TERRITORY_DATA = bucky_cfg["data_dir"] + "/population/territory_pop.csv"
ADD_AMERICAN_SAMOA = False

# CSSE UIDs for Michigan prison information
MI_PRISON_UIDS = [84070004, 84070005]


def get_timeseries_data(col_name, filename, fips_key="FIPS", is_csse=True):
    """Takes a historical data file and reduces it to a dataframe with FIPs, 
    date, and case or death data.
    
    Parameters
    ----------
    col_name : string
        Column name to extract from data.
    filename : string
        Location of filename to read.
    fips_key : string, optional
        Key used in file for indicating county-level field.
    is_csse : boolean
        Indicates whether the file is CSSE data. If True, certain areas
        without FIPS are included.

    Returns
    -------
    df : Pandas DataFrame
        Dataframe with the historical data indexed by FIPS, date

    """

    # Read file
    df = pd.read_csv(filename)

    # CSSE-specific correction
    if is_csse:
        # Michigan prisons have no FIPS, replace with their UID to be processed later
        mi_data = df.loc[df["UID"].isin(MI_PRISON_UIDS)]
        mi_data = mi_data.assign(FIPS=mi_data["UID"])

        df.loc[mi_data.index] = mi_data.values

    # Get dates and FIPS columns only
    cols = list(df.columns)
    idx = cols.index("1/22/20")

    # Keep columns after index
    keep_cols = cols[idx:]

    # Add FIPS
    keep_cols.append(fips_key)

    # Drop other columns
    df = df[keep_cols]

    # Reindex and stack
    df.set_index(fips_key, inplace=True)

    # Stack
    df = df.stack().reset_index()

    # Replace column names
    df.columns = ["FIPS", "date", col_name]

    return df


def distribute_unallocated_csse(confirmed_file, deaths_file, hist_df):
    """Distributes unallocated historical case and deaths data from CSSE.

    JHU CSSE data contains state-level unallocated data, indicated with 
    "Unassigned" or "Out of" for each state. This function distributes 
    these unallocated cases based on the proportion of cases in each
    county relative to the state.
    
    Parameters
    ----------
    confirmed_file : string
        filename of CSSE confirmed data
    deaths_file : string
        filename of CSSE death data
    hist_df : Pandas DataFrame
        current historical DataFrame containing confirmed and death data
        indexed by date and FIPS code

    Returns
    -------
    hist_df : Pandas DataFrame
        modified historical DataFrame with cases and deaths distributed

    """
    hist_df.reset_index(inplace=True)
    if "index" in hist_df.columns:
        hist_df = hist_df.drop(columns=["index"])

    hist_df = hist_df.assign(state_fips=hist_df["FIPS"] // 1000)
    hist_df.set_index(["date", "FIPS"], inplace=True)
    # Read cases and deaths files
    case_df = pd.read_csv(confirmed_file)
    deaths_df = pd.read_csv(deaths_file)

    # Get unassigned and 'out of X'
    cases_unallocated = case_df.loc[
        (case_df["Combined_Key"].str.contains("Out of"))
        | (case_df["Combined_Key"].str.contains("Unassigned"))
    ]
    cases_unallocated = cases_unallocated.assign(
        state_fips=cases_unallocated["FIPS"].astype(str).str[3:].astype(float)
    )

    deaths_unallocated = deaths_df.loc[
        (deaths_df["Combined_Key"].str.contains("Out of"))
        | (deaths_df["Combined_Key"].str.contains("Unassigned"))
    ]
    deaths_unallocated = deaths_unallocated.assign(
        state_fips=deaths_unallocated["FIPS"].astype(str).str[3:].astype(float)
    )

    # Sum unassigned and 'out of X'
    extra_cases = cases_unallocated.groupby("state_fips").sum()
    extra_deaths = deaths_unallocated.groupby("state_fips").sum()
    extra_cases.drop(columns=["UID", "code3", "FIPS", "Lat", "Long_",], inplace=True)
    extra_deaths.drop(
        columns=["UID", "Population", "code3", "FIPS", "Lat", "Long_",], inplace=True
    )

    # Reformat dates to match processed data's format
    extra_cases = extra_cases.rename(
        columns={
            x: datetime.strptime(x, "%m/%d/%y").strftime("%Y-%m-%d")
            for x in extra_cases.columns
        }
    )
    extra_deaths = extra_deaths.rename(
        columns={
            x: datetime.strptime(x, "%m/%d/%y").strftime("%Y-%m-%d")
            for x in extra_deaths.columns
        }
    )

    # Iterate over states in historical data
    for state_fips in tqdm.tqdm(
        extra_cases.index.values,
        desc="Distributing unallocated state data",
        dynamic_ncols=True,
    ):

        # Get extra cases and deaths
        state_extra_cases = extra_cases.xs(state_fips)
        state_extra_deaths = extra_deaths.xs(state_fips)

        # Get historical data
        state_df = hist_df.loc[hist_df["state_fips"] == state_fips]
        state_df = state_df.reset_index()
        state_confirmed = state_df[["FIPS", "date", "cumulative_reported_cases"]]

        state_confirmed = state_confirmed.pivot(
            index="FIPS", columns="date", values="cumulative_reported_cases"
        )
        frac_df = state_confirmed / state_confirmed.sum()
        frac_df = frac_df.replace(np.nan, 0)

        # Distribute cases and deaths based on this matrix
        dist_cases = frac_df.mul(state_extra_cases, axis="columns").T.stack()
        dist_deaths = frac_df.mul(state_extra_deaths, axis="columns").T.stack()

        # Index historical data
        state_df = state_df.set_index(["date", "FIPS"])
        tmp = dist_deaths.to_frame(name="cumulative_deaths")
        tmp["cumulative_reported_cases"] = dist_cases
        state_df += tmp

        hist_df.loc[state_df.index] = state_df.values

    hist_df = hist_df.drop(columns=["state_fips"])
    return hist_df


def distribute_data_by_population(total_df, dist_vect, data_to_dist, replace):
    """Distributes data by population across a state or territory.
    
    Parameters
    ----------
    total_df : Pandas DataFrame
        DataFrame containing confirmed and death data indexed by date and
        FIPS code
    dist_vect : Pandas DataFrame
        Population data for each county as proportion of total state 
        population, indexed by FIPS code
    data_to_dist: Pandas DataFrame
        Data to distribute, indexed by data
    replace : boolean
        If true, distributed values overwrite current historical data in
        DataFrame. If false, distributed values are added to current data


    Returns
    -------
    total_df : Pandas DataFrame
        Modified input dataframe with distributed data

    """

    # Create temporary dataframe and merge
    tmp = total_df.reset_index()
    tmp = tmp.merge(dist_vect, on="FIPS")
    tmp = tmp.merge(data_to_dist, on="date")

    # Use population fraction to scale
    if replace:
        tmp = tmp.assign(
            cumulative_reported_cases=tmp["pop_fraction"]
            * tmp["cumulative_reported_cases_y"]
        )
        tmp = tmp.assign(
            cumulative_deaths=tmp["pop_fraction"] * tmp["cumulative_deaths_y"]
        )
    else:
        tmp = tmp.assign(
            cumulative_reported_cases=tmp["cumulative_reported_cases_x"]
            + tmp["pop_fraction"] * tmp["cumulative_reported_cases_y"]
        )
        tmp = tmp.assign(
            cumulative_deaths=tmp["cumulative_deaths_x"]
            + tmp["pop_fraction"] * tmp["cumulative_deaths_y"]
        )

    # Discard merge columns
    tmp = tmp[["FIPS", "date", "cumulative_reported_cases", "cumulative_deaths"]]
    tmp.set_index(["FIPS", "date"], inplace=True)
    total_df.loc[tmp.index] = tmp.values

    return total_df


def get_county_population_data(csse_deaths_file, county_fips):
    """Uses JHU CSSE deaths file to get county-level population data as 
    as fraction of total population across requested list of counties.
    
    Parameters
    ----------
    csse_deaths_file : string
        filename of CSSE deaths file
    county_fips: array-like
        list of FIPS to return population data for


    Returns
    -------
    population_df: Pandas DataFrame
        DataFrame with population fraction data indexed by FIPS

    """
    # Use CSSE Deaths file to get population values by FIPS
    df = pd.read_csv(csse_deaths_file)
    population_df = df.loc[df["FIPS"].isin(county_fips)][["FIPS", "Population"]]
    population_df.set_index("FIPS", inplace=True)
    population_df = population_df.assign(
        pop_fraction=population_df["Population"] / population_df["Population"].sum()
    )
    population_df.drop(columns=["Population"], inplace=True)

    return population_df


def distribute_nyc_data(df):
    """Distributes NYC case data across the six NYC counties.
    
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing historical data indexed by FIPS and date

    Returns
    -------
    df : Pandas DataFrame
        Modified DataFrame containing corrected NYC historical data 
        indexed by FIPS and date

    """
    # Get population for counties
    nyc_counties = [36005, 36081, 36047, 36085, 36061]

    # CSSE has incorrect population data
    county_populations = {
        36005: 1432132,
        36081: 2278906,
        36047: 2582830,
        36085: 476179,
        36061: 1628701,
    }

    total_nyc_pop = np.sum(list(county_populations.values()))
    population_df = pd.DataFrame(
        data=county_populations.values(),
        columns=["Population"],
        index=county_populations.keys(),
    )
    population_df = population_df.assign(
        pop_fraction=population_df["Population"] / total_nyc_pop
    )
    population_df.drop(columns=["Population"], inplace=True)
    population_df.index.set_names(["FIPS"], inplace=True)

    # All data is in FIPS 36061
    nyc_data = df.xs(36061, level=0)

    df = distribute_data_by_population(df, population_df, nyc_data, True)
    return df


def distribute_mdoc(df, csse_deaths_file):
    """Distributes Michigan Department of Corrections data across Michigan
    counties by population.
    
    Parameters
    ----------
    df : Pandas DataFrame
        Current historical DataFrame indexed by FIPS and date, which
        includes MDOC and FCI data
    csse_deaths_file : string
        File location of CSSE deaths file (contains population data)

    Returns
    -------
    df : Pandas DataFrame
        Modified historical dataframe with Michigan prison data distributed
        and added to Michigan data

    """
    # Get Michigan county populations
    tmp = df.reset_index()
    michigan_counties = tmp.loc[tmp["FIPS"] // 1000 == 26]["FIPS"].unique()
    michigan_pop = get_county_population_data(csse_deaths_file, michigan_counties)

    # Get prison data
    mdoc_data = df.xs(MI_PRISON_UIDS[0], level=0)
    fci_data = df.xs(MI_PRISON_UIDS[1], level=0)

    # Sum and distribute
    mi_unallocated = mdoc_data + fci_data

    # Add to Michigan data, do not replace
    df = distribute_data_by_population(df, michigan_pop, mi_unallocated, False)

    # Drop prison data from dataframe
    df = df.loc[~df.index.get_level_values(0).isin(MI_PRISON_UIDS)]

    return df


def distribute_territory_data(df, add_american_samoa):
    """Distributes territory-wide case and death data for territories.
    
    Uses county population to distribute cases for US Virgin Islands, 
    Guam, and CNMI. Optionally adds a single case to the most populous
    American Samoan county.

    Parameters
    ----------
    df : Pandas DataFrame
        Current historical DataFrame indexed by FIPS and date, which
        includes territory-wide case and death data
    add_american_samoa: boolean
        If true, adds 1 case to American Samoa

    Returns
    -------
    df : Pandas DataFrame
        Modified historical dataframe with territory-wide data 
        distributed to counties

    """
    # Get population data from file
    age_df = pd.read_csv(TERRITORY_DATA, index_col="fips")

    # use age-stratified data to get total pop per county
    pop_df = pd.DataFrame(age_df.sum(axis=1))
    pop_df.reset_index(inplace=True)
    pop_df.rename(columns={"fips": "FIPS", 0: "total"}, inplace=True)

    # Drop PR because CSSE does have county-level PR data now
    pop_df = pop_df.loc[~(pop_df["FIPS"] // 1000).isin([72, 60])]

    # Create nan dataframe for territories (easier to update than append)
    tfips = pop_df["FIPS"].unique()
    dates = df.index.unique(level=1).values
    fips_col = []
    date_col = []
    for fips in tfips:
        for d in dates:
            fips_col.append(fips)
            date_col.append(d)

    tframe = pd.DataFrame.from_dict(
        {
            "FIPS": fips_col,
            "date": date_col,
            "cumulative_reported_cases": [np.nan for d in date_col],
            "cumulative_deaths": [np.nan for d in date_col],
        }
    )
    tframe.set_index(["FIPS", "date"], inplace=True)
    df = df.append(tframe)

    # CSSE has state-level data for Guam, CNMI, USVI
    state_level_fips = [66, 69, 78]

    for state_fips in state_level_fips:
        state_data = df.xs(state_fips, level=0)
        state_pop = pop_df.loc[pop_df["FIPS"] // 1000 == state_fips]
        state_pop = state_pop.assign(
            pop_fraction=state_pop["total"] / state_pop["total"].sum()
        )
        state_pop = state_pop.drop(columns=["total"])
        df = distribute_data_by_population(df, state_pop, state_data, True)

    # Optionally add 1 confirmed case to most populous AS county
    if add_american_samoa:
        as_frame = pd.DataFrame.from_dict(
            {
                "FIPS": [60050 for d in dates],
                "date": dates,
                "cumulative_reported_cases": [1.0 for d in dates],
                "cumulative_deaths": [0.0 for d in dates],
            }
        )
        as_frame.set_index(["FIPS", "date"], inplace=True)
        df = df.append(as_frame)

    return df


def process_csse_data():
    """Performs pre-processing on CSSE data.

    CSSE data is separated into two different files: confirmed cases and
    deaths. These two files are combined into one dataframe, indexed by
    FIPS and date with two columns, Confirmed and Deaths. This function 
    distributes CSSE that is either unallocated or territory-wide instead
    of county-wide. Michigan data from the state Department of Corrections
    and Federal Correctional Institution is distributed to Michigan counties.
    New York City data which is currently all placed in one county (New
    York County) is distributed to the other NYC counties. Territory data
    for Guam, CNMI, and US Virgin Islands is also distributed. This data 
    is written to a CSV.

    """
    data_dir = (
        bucky_cfg["data_dir"]
        + "/cases/COVID-19/csse_covid_19_data/csse_covid_19_time_series/"
    )

    # Get confirmed and deaths files
    confirmed_file = os.path.join(data_dir, "time_series_covid19_confirmed_US.csv")
    deaths_file = os.path.join(data_dir, "time_series_covid19_deaths_US.csv")

    confirmed = get_timeseries_data("Confirmed", confirmed_file)
    deaths = get_timeseries_data("Deaths", deaths_file)

    # rename columns
    confirmed.rename(columns={"Confirmed": "cumulative_reported_cases"}, inplace=True)
    deaths.rename(columns={"Deaths": "cumulative_deaths"}, inplace=True)

    # Merge datasets
    data = pd.merge(confirmed, deaths, on=["FIPS", "date"], how="left").fillna(0)

    # Remove missing FIPS
    data = data[data.FIPS != 0]

    # Replace FIPS with adm2
    # data.rename(columns={"FIPS" : "adm2"}, inplace=True)
    # print(data.columns)

    data = data.set_index(["FIPS", "date"])

    # Distribute territory and Michigan DOC data
    data = distribute_territory_data(data, ADD_AMERICAN_SAMOA)
    data = distribute_mdoc(data, deaths_file)

    data = data.reset_index()
    data = data.assign(date=pd.to_datetime(data["date"]))
    data = data.sort_values(by="date")

    data = distribute_unallocated_csse(confirmed_file, deaths_file, data)

    # Rename FIPS index to adm2
    data.index = data.index.rename(["date", "adm2"])

    # Write to files
    hist_file = bucky_cfg["data_dir"] + "/cases/csse_hist_timeseries.csv"
    logging.info("Saving CSSE historical data as %s" % hist_file)
    data.to_csv(hist_file)


def update_covid_tracking_data():
    """Downloads and processes data from the Atlantic's COVID Tracking project
    to match the format of other preprocessed data sources.

    The COVID Tracking project contains data at a state-level. Each state
    is given a random FIPS selected from all FIPS in that state. This is
    done to make aggregation easier for plotting later. Processed data is
    written to a CSV.

    """
    url = "https://api.covidtracking.com/v1/states/daily.csv"
    filename = bucky_cfg["data_dir"] + "/cases/covid_tracking_raw.csv"
    # Download data
    context = ssl._create_unverified_context()
    # Create filename
    with urllib.request.urlopen(url, context=context) as testfile, open(
        filename, "w"
    ) as f:
        f.write(testfile.read().decode())

    # Read file
    data_file = bucky_cfg["data_dir"] + "/cases/covid_tracking_raw.csv"
    df = pd.read_csv(data_file)

    # Fix date
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    # Rename FIPS
    df.rename(columns={"fips": "adm1"}, inplace=True)

    # Save
    covid_tracking_name = bucky_cfg["data_dir"] + "/cases/covid_tracking.csv"
    logging.info("Saving COVID Tracking Data as %s" % covid_tracking_name)
    df.to_csv(covid_tracking_name, index=False)


def process_usafacts(case_file, deaths_file):
    """Performs preprocessing on USA Facts data.

    USAFacts contains unallocated cases and deaths for each state. These
    are allocated across states based on case distribution in the state.
    
    Parameters
    ----------
    case_file : string
        Location of USAFacts case file
    deaths_file : string
        Location of USAFacts death file

    Returns
    -------
    combined_df : Pandas DataFrame
        USAFacts data containing cases and deaths indexed by FIPS and
        date.

    """
    # Read case file, will be used to scale unallocated cases & deaths
    case_df = pd.read_csv(case_file)
    case_df.drop(columns=["County Name", "State", "stateFIPS"], inplace=True)

    files = [case_file, deaths_file]
    cols = ["Confirmed", "Deaths"]
    processed_frames = []
    for i, file in enumerate(files):

        df = pd.read_csv(file)
        df.drop(columns=["County Name", "State"], inplace=True)

        # Get time series versions
        ts = get_timeseries_data(cols[i], file, "countyFIPS", False)

        ts = ts.loc[~ts["FIPS"].isin([0, 1])]
        ts.set_index(["FIPS", "date"], inplace=True)

        for state_code, state_df in tqdm.tqdm(
            df.groupby("stateFIPS"),
            desc="Processing USAFacts " + cols[i],
            dynamic_ncols=True,
        ):

            # DC has no unallocated row
            if state_df.loc[state_df["countyFIPS"] == 0].empty:
                continue

            state_df.set_index("countyFIPS", inplace=True)
            state_df = state_df.drop(columns=["stateFIPS"])

            # Get unallocated cases
            state_unallocated = state_df.xs(0)
            state_df.drop(index=0, inplace=True)

            if state_code == 36:

                # NYC includes "probable" deaths - these are currently being dropped
                state_df.drop(index=1, inplace=True)

            if state_unallocated.sum() > 0:

                # Distribute by cases
                state_cases = case_df.loc[case_df["countyFIPS"] // 1000 == state_code]
                state_cases.set_index("countyFIPS", inplace=True)
                frac_df = state_cases / state_cases.sum()
                frac_df = frac_df.replace(np.nan, 0)

                # Multiply with unallocated to distribute
                dist_unallocated = frac_df.mul(state_unallocated, axis="columns").T
                state_df = state_df.T + dist_unallocated
                state_df = state_df.T.stack().reset_index()

                # Replace column names
                state_df = state_df.rename(
                    columns={"countyFIPS": "FIPS", "level_1": "date", 0: cols[i]}
                )
                state_df.set_index(["FIPS", "date"], inplace=True)
                ts.loc[state_df.index] = state_df.values

        processed_frames.append(ts)

    # Combine
    combined_df = pd.merge(
        processed_frames[0], processed_frames[1], on=["FIPS", "date"], how="left"
    ).fillna(0)
    return combined_df


def update_usafacts_data():
    """Retrieves updated historical data from USA Facts, preprocesses it,
    and writes to CSV.
    """
    logging.info("Downloading USA Facts data")
    case_url = "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv"
    deaths_url = "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_deaths_usafacts.csv"
    urls = [case_url, deaths_url]

    filenames = [
        bucky_cfg["data_dir"] + "/cases/covid_confirmed_usafacts.csv",
        bucky_cfg["data_dir"] + "/cases/covid_deaths_usafacts.csv",
    ]

    # Download case and death data
    context = ssl._create_unverified_context()
    for i, url in enumerate(urls):

        # Create filename
        with urllib.request.urlopen(url, context=context) as testfile, open(
            filenames[i], "w"
        ) as f:
            f.write(testfile.read().decode())

    # Merge datasets
    data = process_usafacts(filenames[0], filenames[1])
    data = data.reset_index()

    # Sort by date
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(by="date")
    data.rename(columns={"FIPS": "adm2"}, inplace=True)
    data.to_csv(bucky_cfg["data_dir"] + "/cases/usafacts_hist.csv")


def update_repos():
    """Uses git to update public data repos.  
    """
    # Repos to update
    repos = [
        bucky_cfg["data_dir"] + "/cases/COVID-19/",
        bucky_cfg["data_dir"] + "/mobility/DL-COVID-19/",
        bucky_cfg["data_dir"] + "/mobility/COVIDExposureIndices/",
    ]

    for repo in repos:
        git_pull(repo)

    # Process CSSE data
    process_csse_data()

    # Process COVID Tracking Data
    update_covid_tracking_data()

    # Process USA Facts
    update_usafacts_data()


def git_pull(abs_path):
    """Updates a git repository given its path.
    
    Parameters
    ----------
    abs_path : string
        Abs path location of repository to update
    """

    git_command = "git pull --rebase origin master"

    # pull
    process = subprocess.Popen(
        git_command.split(), stdout=subprocess.PIPE, cwd=abs_path
    )
    output, error = process.communicate()

    if error:

        # Get name of repo
        git_name = "git remote -v"
        process = subprocess.Popen(git_name.split(), stdout=subprocess.PIPE)
        logging.error("Error pulling from repo: " + output)


if __name__ == "__main__":
    update_repos()
