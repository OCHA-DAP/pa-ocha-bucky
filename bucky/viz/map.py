import argparse
import glob
import logging
import os
import sys
from datetime import timedelta

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatter, ScalarFormatter
from tqdm import tqdm

from ..util.read_config import bucky_cfg
from ..util.readable_col_names import readable_col_names
from ..util.util import remove_chars
from .geoid import read_geoid_from_graph, read_lookup

parser = argparse.ArgumentParser(description="Bucky model mapping tools")

# Location of processed data
parser.add_argument(
    "-i",
    "--input_dir",
    default=max(
        glob.glob(os.path.join(bucky_cfg["output_dir"], "*/")),
        key=os.path.getctime,
        default="Most recently created folder in output_dir",
    ),
    type=str,
    help="Directory location of processed simulation data",
)

# Output directory
parser.add_argument(
    "-o",
    "--output",
    default=None,
    type=str,
    help="Output directory for maps. Defaults to input_dir/maps/",
)

# Graph file used for this run. Defaults to most recently created
parser.add_argument(
    "-g", "--graph_file", default=None, type=str, help="Graph file used during model",
)

# Data columns
default_plot_cols = ["daily_cases_reported", "daily_deaths"]
parser.add_argument(
    "--columns",
    default=default_plot_cols,
    nargs="+",
    type=str,
    help="Data columns to plot. Maps are created separately for each requested column",
)

# Use mean or median values (median default)
parser.add_argument(
    "--mean",
    action="store_true",
    help="Use mean value instead of median value for map",
)

# Use log or linear (log default)
parser.add_argument(
    "--linear",
    action="store_true",
    help="Use linear scaling for values instead of log",
)

# Frequency of maps (default weekly)
parser.add_argument(
    "-f",
    "--freq",
    default="weekly",
    choices=["daily", "weekly", "monthly"],
    help="Frequency at which to create maps",
)

# Optionally pass in specific dates (takes priority over frequency)
parser.add_argument(
    "-d", "--dates", default=None, type=str, nargs="+", help="Specific dates to map"
)

# Create country-level plot
parser.add_argument("--adm0", action="store_true", help="Create adm0-level plot ")

# Create plots for every admin1
parser.add_argument(
    "--all_adm1",
    action="store_true",
    help="Create adm1-level plot for every available adm1-level area",
)

# Specify admin1
parser.add_argument(
    "--adm1",
    default=None,
    type=str,
    nargs="+",
    help="Create adm1-level plot for the requested adm1 name",
)

# Shape file information
parser.add_argument(
    "--adm1_shape",
    default=os.path.join(bucky_cfg["data_dir"], "shapefiles/tl_2019_us_state.shp"),
    type=str,
    help="Location of admin1 shapefile",
)
parser.add_argument(
    "--adm2_shape",
    default=os.path.join(bucky_cfg["data_dir"], "shapefiles/tl_2019_us_county.shp"),
    type=str,
    help="Location of admin2 shapefile",
)

# Shape file information
parser.add_argument(
    "--adm1_col", default="STATEFP", type=str, help="Shapefile adm1 column name",
)

parser.add_argument(
    "--adm2_col", default="GEOID", type=str, help="Shapefile adm2 column name",
)

# Can pass in a lookup table to use in place of graph
parser.add_argument(
    "--lookup", default=None, type=str, help="Lookup table for geographic mapping info",
)

# Colormap
default_cmap = "Reds"
parser.add_argument(
    "-c",
    "--cmap",
    default=default_cmap,
    type=str,
    help="Colormap to use. Must be a valid matplotlib colormap.",
)


def get_dates(df, frequency="weekly"):
    """Given a DataFrame of simulation data, this method returns dates 
    based on the requested frequency. 

    
    Parameters
    ----------
    df : Pandas DataFrame
        Dataframe of simulation data
    frequency : {'daily', 'monthly', 'weekly' (default)}
        Frequency of selected dates

    Returns
    -------
    date_list : list of strings
        List of dates

    """
    if frequency == "daily":
        interval = 1

    elif frequency == "monthly":
        interval = 28

    else:
        interval = 7

    # Get dates from dateframe
    dates = df["date"]

    # cast to datetime objects
    dates = pd.to_datetime(dates)

    # Create list of dates
    date_list = []

    # Get start date of simulation
    start = dates.min()
    date_list.append(start)

    # Iterate by interval until end is reached
    next_date = start
    while next_date < dates.max():
        next_date = next_date + timedelta(days=interval)

        if next_date <= dates.max():
            date_list.append(next_date)

    return date_list


def get_map_data(data_dir, adm_level, use_mean=False):
    """Reads requested simulation data.
    
    Maps are created using one level down from the requested map level.
    For example, a national map is created using state-level data.

    Parameters
    ----------
    data_dir : string
        Location of preprocessed simulation data
    adm_level : {'adm0', adm1'}
        Admin level of requested map
    use_mean : boolean
        If true, uses mean data. Otherwise, uses median quantile

    Returns
    -------
    df : Pandas DataFrame
        Requested preprocessed simulation data

    """
    # Determine filename
    file_prefix = "adm1" if adm_level == "adm0" else "adm2"

    if use_mean:

        # Read file
        filename = os.path.join(input_dir, file_prefix + "_mean_std.csv")
        df = pd.read_csv(filename)

        # Keep mean only
        df = df.loc[df["stat"] == "mean"]

    else:

        # Read file
        filename = os.path.join(input_dir, file_prefix + "_quantiles.csv")
        df = pd.read_csv(filename)

        # Keep median
        df = df.loc[df["quantile"] == 0.5]

    return df


def make_map(
    shape_df,
    df,
    dates,
    adm_key,
    cols,
    output_dir,
    title_prefix=None,
    log_scale=True,
    colormap="Reds",
    outline_df=None,
):
    """Creates a map for each date and column.
    
    Parameters
    ----------
    shape_df : Geopandas GeoDataFrame
        Shapefile information at the required admin level
    df : Pandas DataFrame
        Simulation data to plot
    dates : list of strings
        List of dates to make maps for
    cols : list of strings
        List of columns to make maps for
    output_dir : string
        Directory to place created maps
    title_prefix : string or None
        String to add to map prefix
    log_scale : boolean
        If true, uses log scaling
    colormap : string (default: 'Reds')
        Colormap to use; must be a valid Matplotlib colormap
    outline_df : Geopandas GeoDataFrame or None
        Shapefile for outline
    """

    # Maps are joined one level down from the map-level
    # National maps color by state, state maps color by county
    join_key = "adm2" if adm_key == "adm1" else "adm1"

    # For colorbar scaling
    data_ranges = []

    # If log, use a different colormap and scale columns
    if log_scale:
        formatter = LogFormatter(10, labelOnlyBase=False)

        # Log scale columns and save
        for col in cols:

            col_name = "log_" + col
            df = pd.concat([df, np.log10(1.0 + df[col]).rename(col_name)], axis=1)

            # Save min and max
            data_ranges.append([df[col_name].min(), df[col_name].max()])

    else:
        formatter = ScalarFormatter()

        for col in cols:
            data_ranges.append([df[col].min(), df[col].max()])

    # Index by date
    df["date"] = pd.to_datetime(df["date"])
    df.set_index(["date", join_key], inplace=True)

    # Make maps for each requested data
    for date in dates:

        # Get data for this date and merge with shape data
        date_df = shape_df.merge(df.xs(date, level=0), on=join_key)

        for i, column in enumerate(cols):

            # Create title
            map_title = readable_col_names[column] + " " + str(date.date())

            if title_prefix is not None:
                map_title = title_prefix + " " + map_title
            val_range = data_ranges[i]

            # Determine column to plot
            if log_scale:
                column = "log_" + column

            # date_df = date_df.to_crs(epsg=2163)

            # Plot
            ax = date_df.plot(
                column=column,
                cmap=colormap,
                vmin=val_range[0],
                vmax=val_range[1],
                legend=True,
                legend_kwds={
                    "orientation": "horizontal",
                    "shrink": 0.8,
                    "aspect": 50,
                    "label": "Log10-scaled (10 ^ x)",
                },
                edgecolor="black",
                linewidth=0.1,
                figsize=(16, 9),
            )

            if outline_df is not None:
                outline_df.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)

            plt.title(map_title)
            plt.tight_layout()
            plt.axis("off")
            axes = plt.gca()
            axes.set_aspect("equal")

            # Create filename
            filename = os.path.join(
                output_dir, adm_key + "_" + map_title.replace(" ", "") + ".png"
            )

            # print(filename)
            # plt.show()
            df.xs(date, level=0).to_csv(
                os.path.join(
                    output_dir, adm_key + "_" + map_title.replace(" ", "") + ".csv"
                )
            )
            plt.savefig(filename)
            plt.close()


def get_state_outline(adm2_data, adm1_data):
    """Given admin2 shape data, finds matching admin1 shape data in order
    to get the admin1 outline. 
    
    Parameters
    ----------
    adm2_data : Geopandas GeoDataFrame
        Admin2-level shape data

    adm1_data : Geopandas GeoDataFrame
        Admin1-level shape data

    Returns
    -------
    outline_df : Geopandas GeoDataFrame
        Admin1-level shape data that match values in admin2

    """
    adm1_vals = adm2_data["adm1"].unique()

    outline_df_full = adm1_data.loc[adm1_data["adm1"].isin(adm1_vals)]

    outline_df = gpd.overlay(adm2_data, outline_df_full, how="intersection")

    return outline_df


def make_adm1_maps(
    adm2_shape_df,
    adm1_shape_df,
    df,
    lookup_df,
    dates,
    cols,
    adm1_list,
    output_dir,
    log_scale=True,
    colormap="Reds",
    add_outline=False,
):
    """Creates adm1 maps.
    
    Parameters
    ----------
    adm2_shape_df : Geopandas GeoDataFrame
        Shapefile information at the admin2 level
    adm1_shape_df : Geopandas GeoDataFrame
        Shapefile information at the admin1 level
    df : Pandas DataFrame
        Simulation data to plot
    lookup_df : Pandas DataFrame
        Dataframe containing mapping between admin levels
    dates : list of strings
        List of dates to make maps for
    cols : list of strings
        List of columns to make maps for
    adm1_list : list of strings or None
        List of explicit admin1 names to create names for. If None, a map 
        is made for each unique admin1 in the lookup table
    output_dir : string
        Directory to place created maps
    log_scale : boolean (default: True)
        If true, uses log scaling
    colormap : string, (default: 'Reds')
        Colormap to use; must be a valid Matplotlib colormap
    add_outline : boolean (default: False)
        Add a thicker outline to the map

    """
    # make subdir for adm1 maps
    map_dir = os.path.join(output_dir, "ADM1")

    # create directory
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)

    # Optional dataframe with the state outline
    state_outline_df = None

    if adm1_list is None:

        # Get all adm1 names
        adm1_list = lookup_df["adm1_name"].unique()

    # Iterate over list and make a map for each
    for admin_area in tqdm(adm1_list, total=len(adm1_list)):

        # Find admin 2 values in this admin1
        admin1_code = lookup_df.loc[lookup_df["adm1_name"] == admin_area][
            "adm1"
        ].unique()[0]
        admin2_vals = lookup_df.loc[lookup_df["adm1_name"] == admin_area][
            "adm2"
        ].unique()

        area_data = df.loc[df["adm2"].isin(admin2_vals)]
        area_data = area_data.assign(adm1=admin1_code)

        # Only keep shape data that matches
        # area_shape = shape_df.loc[shape_df['adm1'] == admin1_code ]
        area_shape = adm2_shape_df.loc[adm2_shape_df["adm2"].isin(admin2_vals)]

        # If adding outline, get matching state shapes
        if add_outline:
            state_outline_df = get_state_outline(area_shape, adm1_shape_df)

        make_map(
            shape_df=area_shape,
            df=area_data,
            dates=dates,
            adm_key="adm1",
            cols=cols,
            output_dir=map_dir,
            title_prefix=admin_area,
            log_scale=log_scale,
            colormap=colormap,
            outline_df=state_outline_df,
        )


if __name__ == "__main__":

    # Parse CLI args
    args = parser.parse_args()

    # Check all required parameters are passed in depending on requested maps
    if args.adm0 and args.adm1_shape is None:

        logging.error("Error: ADM1 shapefiles are required for ADM0 maps.")
        sys.exit()

    if (args.adm1 or args.all_adm1) and args.adm2_shape is None:
        logging.error("Error: ADM2 shapefiles are required for ADM1 maps.")
        sys.exit()

    # Parse other arguments
    input_dir = args.input_dir

    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.join(input_dir, "maps")

    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get colormap
    cmap = args.cmap

    # Make sure its a valid matplotlib colormap
    if cmap != default_cmap:

        if cmap not in plt.colormaps():

            logging.error(
                "Error: "
                + cmap
                + " is not a valid matplotlib colormap. Defaulting to: "
                + default_cmap
            )
            cmap = default_cmap

    map_cols = args.columns
    use_mean = args.mean
    dates = args.dates
    use_log = False if args.linear else True
    adm1_col_name = args.adm1_col
    adm2_col_name = args.adm2_col

    # Create maps
    if args.adm0:

        # Get data
        df = get_map_data(input_dir, "adm0", use_mean)

        # Get dates
        if dates is None:
            dates = get_dates(df, args.freq)

        # Read adm1 shapefile
        shape_data = gpd.read_file(args.adm1_shape)

        # Rename join column
        if adm1_col_name != "adm1":
            shape_data = shape_data.rename(columns={adm1_col_name: "adm1"})

        # Column management - adm1/2 vals should be integers only
        shape_data["adm1"] = shape_data["adm1"].apply(remove_chars).astype(int)

        # Send to map
        make_map(
            shape_df=shape_data,
            df=df,
            dates=dates,
            adm_key="adm0",
            cols=map_cols,
            output_dir=output_dir,
            title_prefix="ADM0",
            log_scale=use_log,
            colormap=cmap,
        )

    if args.adm1 or args.all_adm1:

        if args.lookup is not None:
            lookup_df = read_lookup(args.lookup)
            add_state_outline = True
        else:
            lookup_df = read_geoid_from_graph(args.graph_file)
            add_state_outline = True

        # Get data
        df = get_map_data(input_dir, "adm1", use_mean)

        # Get dates
        if dates is None:
            dates = get_dates(df, args.freq)

        # Read adm1 shapefile
        adm2_shape_data = gpd.read_file(args.adm2_shape)

        # Rename join column
        if adm1_col_name != "adm1":
            adm2_shape_data = adm2_shape_data.rename(columns={adm1_col_name: "adm1"})

        # Cast to int
        adm2_shape_data["adm1"] = (
            adm2_shape_data["adm1"].apply(remove_chars).astype(int)
        )
        adm2_shape_data = adm2_shape_data.assign(
            adm2=adm2_shape_data[adm2_col_name].apply(remove_chars).astype(int)
        )

        # use adm1 shape data as well
        adm1_shape_data = gpd.read_file(args.adm1_shape)

        # Rename join column
        if adm1_col_name != "adm1":
            adm1_shape_data = adm1_shape_data.rename(columns={adm1_col_name: "adm1"})

        # Cast to int
        adm1_shape_data["adm1"] = (
            adm1_shape_data["adm1"].apply(remove_chars).astype(int)
        )

        make_adm1_maps(
            adm2_shape_df=adm2_shape_data,
            adm1_shape_df=adm1_shape_data,
            df=df,
            lookup_df=lookup_df,
            dates=dates,
            cols=map_cols,
            adm1_list=args.adm1,
            output_dir=output_dir,
            log_scale=use_log,
            colormap=cmap,
            add_outline=add_state_outline,
        )
