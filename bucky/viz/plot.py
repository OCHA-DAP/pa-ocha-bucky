import argparse
import glob
import logging
import os
import pickle
import sys

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
import tqdm

from ..util.get_historical_data import get_historical_data
from ..util.read_config import bucky_cfg
from ..util.readable_col_names import readable_col_names
from .geoid import read_geoid_from_graph, read_lookup

import matplotlib  # isort:skip

matplotlib.rc("axes.formatter", useoffset=False)  # isort:skip
import matplotlib.pyplot as plt  # isort:skip
from matplotlib.ticker import StrMethodFormatter  # isort:skip


plt.style.use("ggplot")

parser = argparse.ArgumentParser(description="Bucky model plotting tools")

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
    help="Directory location of aggregated data",
)

# Output directory
parser.add_argument(
    "-o",
    "--output",
    default=None,
    type=str,
    help="Output directory for plots. Defaults to input_dir/plots/",
)

# Graph file used for this run. Defaults to most recently created
parser.add_argument(
    "-g",
    "--graph_file",
    default=None,
    type=str,
    help="Graph file used during model. Defaults to most recently created graph",
)

# Aggregation levels, e.g. state, county, etc.
parser.add_argument(
    "-l",
    "--levels",
    default=["adm0", "adm1"],
    nargs="+",
    type=str,
    help="Requested plot levels",
)

# Columns for plot, historical data
default_plot_cols = ["daily_reported_cases", "daily_deaths"]

parser.add_argument(
    "--plot_columns",
    default=default_plot_cols,
    nargs="+",
    type=str,
    help="Columns to plot",
)

# Can pass in a lookup table to use in place of graph
parser.add_argument(
    "--lookup", default=None, type=str, help="Lookup table for geographic mapping info",
)

# Pass in the minimum number of historical data points to plot
parser.add_argument(
    "--min_hist",
    default=0,
    type=int,
    help="Minimum number of historical data points to plot.",
)

# Pass in a specific historical start date and historical file
parser.add_argument(
    "--hist_start",
    default=None,
    type=str,
    help="Start date of historical data. If not passed in, will align with start date of simulation",
)

# Optional flags
parser.add_argument(
    "--adm1_name", default=None, type=str, help="Admin1 to make admin2-level plots for",
)

parser.add_argument(
    "--end_date",
    default=None,
    type=str,
    help="Data will not be plotted past this point",
)

parser.add_argument(
    "-v", "--verbose", action="store_true", help="Print extra information"
)

parser.add_argument(
    "-hist",
    "--hist",
    action="store_true",
    help="Plot historical data in addition to simulation data",
)

parser.add_argument(
    "--hist_file",
    type=str,
    default=None,
    help="Path to historical data file. If None, uses either CSSE or Covid Tracking data depending on columns requested.",
)

parser.add_argument(
    "-q",
    "--quantiles",
    nargs="+",
    type=float,
    default=None,
    help="Specify the quantiles to plot. Defaults to all quantiles present in data.",
)

# Size of window in days
parser.add_argument(
    "-w",
    "--window_size",
    default=7,
    type=int,
    help="Size of window (in days) to apply to historical data",
)


def interval(mean, sem, conf, N):
    z = scipy.stats.t.ppf((1 + conf) / 2.0, N - 1)
    return (mean - sem * z).clip(lower=0.0), mean + sem * z


def plot(
    output_dir, lookup_df, key, sim_data, hist_data, plot_columns, quantiles,
):
    """Given a dataframe and a key, creates plots with requested columns.

    For example, a DataFrame with state-level data would create a plot for
    each unique state. Simulation data is plotted as a line with shaded
    confidence intervals. Historical data is added as scatter points if
    requested. 

    Parameters
    ----------
    output_dir : string
        Location to place created plots.
    lookup_df : Pandas DataFrame
        Dataframe containing information relating different geographic
        areas
    key : string
        Key to use to relate simulation data and geographic areas. Must
        appear in lookup and simulation data (and historical data if 
        applicable)
    sim_data : Pandas DataFrame
        Simulation data to plot
    hist_data : Pandas DataFrame
        Historical data to add to plot
    plot_columns : list of strings
        Columns to plot
    quantiles : list of floats (or None)
        List of quantiles to plot. If None, will plot all available 
        quantiles in data.
    """

    # If quantiles were not specified, get all quantiles present in data
    if quantiles is None:
        quantiles = sim_data["quantile"].unique()

    # Get number of quantiles
    num_intervals = len(quantiles)

    # make sure they're sorted
    quantiles.sort()

    # Drop lookup nans
    lookup_df = lookup_df.dropna()

    # Make one plot for each area
    # For state plots, one plot per state
    # For counties, one plot per county
    unique_sim_areas = sim_data[key].unique()
    unique_lookup_areas = lookup_df[key].unique()

    # Some lookups have a subset. Use whichever set of areas is smaller
    if len(unique_sim_areas) < len(unique_lookup_areas):
        unique_areas = unique_sim_areas
    else:
        unique_areas = unique_lookup_areas

    for area in tqdm.tqdm(
        unique_areas,
        total=len(unique_areas),
        desc="Plotting " + key,
        dynamic_ncols=True,
    ):

        # Get name
        name = lookup_df.loc[lookup_df[key] == area][key + "_name"].values[0]

        # If admin2-level, get admin1-level as well
        # TODO change to function call with descriptive name
        if key == "adm2":

            admin1_name = lookup_df.loc[lookup_df[key] == area]["adm1_name"].values[0]
            name = admin1_name + " : " + name

        # Get data for this area
        area_data = sim_data.loc[sim_data[key] == area]
        area_data = area_data.assign(date=pd.to_datetime(area_data["date"]))

        # Initialize figure
        fig, axs = plt.subplots(
            2,
            sharex=True,
            gridspec_kw={"hspace": 0.1, "height_ratios": [2, 1]},
            figsize=(15, 7),
        )
        fig.suptitle(name.upper())

        for i, col in enumerate(plot_columns):

            # Set index
            area_data.set_index(["date", "quantile"], inplace=True)

            # Middle is median
            median_data = area_data.xs(quantiles[int(num_intervals / 2)], level=1)[col]
            dates = median_data.index.values

            # Plot median and outer quantiles
            median_data.plot(
                linewidth=1.5,
                color="k",
                alpha=0.75,
                label=readable_col_names[col],
                ax=axs[i],
            )

            # Iterate over pairs of quantiles
            num_quantiles = len(quantiles)

            # Scale opacity
            alpha = 1.0 / (num_quantiles // 2)
            for q in range(num_quantiles // 2):

                lower_q = quantiles[q]
                upper_q = quantiles[num_quantiles - 1 - q]

                lower_data = area_data.xs(lower_q, level=1)[col]
                upper_data = area_data.xs(upper_q, level=1)[col]

                axs[i].fill_between(
                    dates,
                    lower_data,
                    upper_data,
                    linewidth=0,
                    alpha=alpha,
                    color="b",
                    interpolate=True,
                )

            # axs[i].set_ylim([.8*oq_lower.min(), 1.2*oq_upper.max()])
            area_data.reset_index(inplace=True)

            # Plot historical data which is already at the correct level
            if hist_data is not None and i <= (len(plot_columns) - 1):

                actuals = hist_data.loc[hist_data[key] == area]
                if not actuals.empty and plot_columns[i] in hist_data.columns:

                    actuals = actuals.assign(date=pd.to_datetime(actuals["date"]))
                    # actuals.set_index('date', inplace=True)
                    hist_label = "Historical " + readable_col_names[plot_columns[i]]
                    actuals.plot(
                        x="date",
                        y=plot_columns[i],
                        ax=axs[i],
                        color="r",
                        marker="o",
                        ls="",
                        label=hist_label,
                    )

                    # Set xlim
                    axs[i].set_xlim(actuals["date"].min(), dates.max())
                    # axs[i].scatter(actual_dates[ind:], actual_vals[ind:], label='Historical data', color='b')
                else:
                    logging.warning(
                        "Historical data missing for area: "
                        + name
                        + ", column="
                        + plot_columns[i]
                    )

            axs[i].grid(True)
            axs[i].legend()
            axs[i].set_ylabel("Count")
            # axs[i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

        plot_filename = os.path.join(
            output_dir, readable_col_names[plot_columns[0]] + "_" + name + ".png"
        )
        plot_filename = plot_filename.replace(" : ", "_")
        plot_filename = plot_filename.replace(" ", "")
        plt.savefig(plot_filename)
        plt.close()

        # Save CSV
        csv_filename = os.path.join(output_dir, name + ".csv")
        area_data.to_csv(csv_filename)


def make_plots(
    adm_levels,
    input_dir,
    output_dir,
    lookup,
    plot_hist,
    plot_columns,
    quantiles,
    window_size,
    end_date,
    hist_file,
    min_hist_points,
    admin1=None,
    hist_start=None,
):
    """Wrapper function around plot. Creates plots, aggregating data 
    if necessary.

    Parameters
    ----------
    adm_levels : list of strings
        List of ADM levels to make plots for
    input_dir : string
        Location of simulation data
    output_dir : string
        Parent directory to place created plots. 
    lookup : Pandas DataFrame
        Lookup table for geographic mapping information
    plot_hist : boolean
        If true, will plot historical data
    plot_columns : list of strings
        List of columns to plot from data
    quantiles : list of floats (or None)
        List of quantiles to plot. If None, will plot all available 
        quantiles in data.
    window_size : int
        Size of window (in days) to apply to historical data
    end_date : string, formatted as YYYY-MM-DD
        Plot data until this date. If None, uses last date in simulation
    hist_file : string
        Path to historical data file. If None, uses either CSSE or
        Covid Tracking data depending on columns requested.
    min_hist_points : int
        Minimum number of historical data points to plot.
    admin1 : list of strings, or None
        List of admin1 values to make plots for. If None, a plot will be
        created for every unique admin1 values. Otherwise, plots are only
        made for those requested.
    hist_start : string, formatted as YYYY-MM-DD
        Plot historical data from this point. If None, aligns with
        simulation start date
    """

    # Loop over requested levels
    for level in adm_levels:

        filename = level + "_quantiles.csv"

        # Read the requested file from the input dir
        data = pd.read_csv(os.path.join(input_dir, filename))
        data = data.assign(date=pd.to_datetime(data["date"]))

        # Get dates for data
        start_date = data["date"].min()
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
        else:
            end_date = data["date"].max()

        # Drop data not within range
        data = data.loc[(data["date"] < end_date) & (data["date"] > start_date)]

        # Determine if sub-folder is necessary (state or county-level)
        plot_dir = output_dir
        if level == "adm2" or level == "adm1":

            plot_dir = os.path.join(plot_dir, level.upper())

            # create directory
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

        # For admin2 only: if a admin1 name was passed in, only keep data within that admin1
        if level == "adm2" and admin1 is not None:

            admin2_vals = lookup_df.loc[lookup_df["adm1_name"] == admin1][
                "adm2"
            ].unique()
            data = data.loc[data["adm2"].isin(admin2_vals)]

        # Read historical data if needed
        hist_data = None
        if plot_hist:

            # Get historical data for requested columns
            hist_data = get_historical_data(
                plot_columns, level, lookup_df, window_size, hist_file
            )

            # Check if historical data was not successfully fetched
            if hist_data is None:
                logging.warning(
                    "No historical data could be found for: " + str(hist_columns)
                )

            else:
                hist_data.reset_index(inplace=True)

                # Drop data not within requested time range
                hist_data = hist_data.assign(date=pd.to_datetime(hist_data["date"]))

                if hist_start is not None:
                    start_date = hist_start

                # Check that there are the minimum number of points
                last_hist_date = hist_data["date"].max()
                num_points = (last_hist_date - pd.to_datetime(start_date)).days

                # Shift start date if necessary
                if num_points < min_hist_points:
                    start_date = last_hist_date - pd.Timedelta(
                        str(min_hist_points) + " days"
                    )

                hist_data = hist_data.loc[
                    (hist_data["date"] < end_date) & (hist_data["date"] > start_date)
                ]
        plot(
            output_dir=plot_dir,
            lookup_df=lookup_df,
            key=level,
            sim_data=data,
            hist_data=hist_data,
            plot_columns=plot_columns,
            quantiles=quantiles,
        )


if __name__ == "__main__":

    # Logging
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
    )

    # Parse CLI args
    args = parser.parse_args()
    logging.info(args)

    # Parse arguments
    input_dir = args.input_dir

    output_dir = args.output

    if output_dir is None:
        output_dir = os.path.join(input_dir, "plots")

    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # aggregation levels
    levels = args.levels

    # Columns for plotting
    plot_cols = args.plot_columns

    if args.lookup is not None:
        lookup_df = read_lookup(args.lookup)
    else:
        lookup_df = read_geoid_from_graph(args.graph_file)

    # Historical data start
    hist_start = args.hist_start

    # Parse optional flags
    window = args.window_size
    plot_historical = args.hist
    verbose = args.verbose
    end_date = args.end_date
    quantiles = args.quantiles
    hist_file = args.hist_file
    min_hist = args.min_hist

    # If a historical file was passed in, make sure hist is also true
    if hist_file is not None:
        plot_historical = True

    # Plot
    make_plots(
        levels,
        input_dir,
        output_dir,
        lookup_df,
        plot_historical,
        plot_cols,
        quantiles,
        window,
        end_date,
        hist_file,
        min_hist,
        args.adm1_name,
        hist_start,
    )
