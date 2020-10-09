import logging
import os
import numpy as np
import pandas as pd

from read_config import bucky_cfg

covid_tracking = os.path.join(bucky_cfg["data_dir"],"cases/covid_tracking.csv")
csse = os.path.join(bucky_cfg["data_dir"],"cases/csse_hist_timeseries.csv")

# Specify file and column name
data_locations = {
    "cumulative_cases": {"file": csse , "column": "Confirmed"},
    "cumulative_deaths": {"file": csse, "column": "Deaths"},
    "hospitalizations": {"file": covid_tracking, "column": "hospitalizedCurrently"},
    "daily_cases_reported": {"file": csse, "column": "Confirmed_daily"},
    "daily_cases": {"file": csse, "column": "Confirmed_daily"},
    "daily_deaths": {"file": csse, "column": "Deaths_daily"},
    "VENT": {"file": covid_tracking, "column": "onVentilatorCurrently"},
    "ICU": {"file": covid_tracking, "column": "inIcuCurrently"},
    "NH": {"file": covid_tracking, "column": "hospitalizedIncrease"},
}

def add_daily_history(history_data, window_size=None):
    """Applies a window to cumulative historical data to get daily data.
    
    Parameters
    ----------
    history_data : Pandas DataFrame
        Cumulative case and death data
    window_size : int or None
        Size of window in days

    Returns
    -------
    history_data : Pandas DataFrame
        Historical data with added columns for daily case and death data
    """
    # TODO: Correct territory data by distributing
    # history_data['adm2'].replace(66, 66010, inplace=True)
    # history_data['adm2'].replace(69, 69110, inplace=True)

    history_data = history_data.set_index(["adm2", "date"])
    history_data.sort_index(inplace=True)

    # Remove string columns if they exist
    str_cols = list(history_data.select_dtypes(include=["object"]).columns)
    history_data = history_data.drop(columns=str_cols)

    daily_data = history_data.groupby(level=0).diff()
    daily_data.columns = [str(col) + "_daily" for col in daily_data.columns]

    if window_size is not None:

        daily_data = (
            daily_data.reset_index(level=0)
            .groupby("adm2")
            .rolling(window_size, min_periods=window_size // 2, center=True)
            .mean()
            .drop(columns=["adm2"])
        )

        # daily_data.reset_index().set_index('adm2', inplace=True)
        # daily_data = daily_data.rolling(7, center=True, on='date').mean()
        # daily_data = hdaily_data.set_index(['adm2', 'date'])

    history_data = history_data.merge(daily_data, left_index=True, right_index=True)
    history_data.reset_index(inplace=True)

    # TODO Set negative values to 0

    return history_data

def get_historical_data(columns, level, lookup_df, window_size):
    """Gets historical data for the columns requested. 
    
    Parameters
    ----------
    columns : list of str
        Column names for historical data
    level : str
        Geographic level to get historical data for, e.g. adm1
    lookup_df : Pandas DataFrame
        Dataframe with names and values for admin0, admin1, and admin2
        levels
    window_size : int
        Size of window in days

    Returns
    -------
    history_data : Pandas DataFrame
        Historical data indexed by data and geographic level 
        containing only requested columns
    """
    df = None

    for requested_col in columns:
        
        # Get file and column name
        file = data_locations[requested_col]["file"]
        column_name = data_locations[requested_col]["column"]

        # Read file
        data = pd.read_csv(file, na_values=0.)
        
        # Add daily history if requested daily deaths or daily cases
        daily_cols = ["daily_cases", "daily_cases_reported", "daily_deaths"]
        if requested_col in daily_cols:
            data = add_daily_history(data, window_size)

        if level == "adm2" and "adm2" not in data.columns:
            logging.warning("ADM2-level data is not available for: " + requested_col)
        
        else:

            # If data doesn't have a column corresponding to requested 
            # level, use lookup table to add it
            if level not in data.columns:

                # Use either county or state to aggregate as needed
                lookup_col = "adm2" if "adm2" in data.columns else "adm1"
                lookup_df.set_index(lookup_col, inplace=True)
                level_dict = lookup_df[level].to_dict()
                data[level] = data[lookup_col].map(level_dict)    

                # Drop items that were not mappable
                data = data.dropna(subset=[level])

            # Aggregate on geographic level
            agg_data = data.groupby(["date", level]).sum()
            agg_data = agg_data.rename(columns={column_name : requested_col})

            # If first column, initialize dataframe
            if df is None:
                df = agg_data[requested_col].to_frame()
            # Combine with previous data
            else:
                df = df.merge(agg_data[requested_col].to_frame(), how="outer", left_index=True, right_index=True)

        lookup_df.reset_index(inplace=True)

    return df
        

if __name__ == "__main__":

    from bucky.viz.geoid import read_geoid_from_graph
    look = read_geoid_from_graph('data/input_graphs/usa--2020-10-06__16_25_10.p') 
    #look.set_index("adm2", inplace=True)
    levels = ['adm0']

    cols = ["VENT", "daily_cases"]

    for level in levels:

        df = get_historical_data(cols, level, look, 7)







