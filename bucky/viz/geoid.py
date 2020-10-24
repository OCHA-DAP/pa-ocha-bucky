import glob
import logging
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd

from ..util import remove_chars
from ..util.read_config import bucky_cfg


def read_geoid_from_graph(graph_file=None):
    """Creates a dataframe relating geographic administration levels, 
    e.g. admin2 values in a given admin1.

    Parameters
    ----------
    graph_file : string or None
        Location of graph file. If None, uses most recently created graph
        in data/input_graphs/

    Returns
    -------
    df : Pandas DataFrame
        Dataframe with names and values for admin0, admin1, and admin2
        levels

    """
    # Use most recently created graph file if one was not provided
    if graph_file is None:
        graph_file = max(
            glob.glob(os.path.join(bucky_cfg["data_dir"], "input_graphs/*.p")),
            key=os.path.getctime,
        )

    logging.info(f"Using {graph_file} for admin level mappings.")

    with open(graph_file, "rb") as f:
        G = pickle.load(f)

    # Get admin0 name
    admin0_name = G.graph["adm0_name"]

    # Get mapping of admin1 code to admin1 name
    admin1_name_map = G.graph["adm1_to_str"]

    # Get admin1 values
    admin1_vals = np.fromiter(
        [remove_chars(x) for x in nx.get_node_attributes(G, G.graph["adm1_key"]).values()], dtype=int
    )

    # Get admin2 values
    admin2_vals = np.fromiter(
        [remove_chars(x) for x in nx.get_node_attributes(G, G.graph["adm2_key"]).values()], dtype=int
    )
    admin2_names = list(nx.get_node_attributes(G, "adm2_name").values())

    # Create initial dataframe
    df = pd.DataFrame(np.vstack((admin2_vals, admin2_names, admin1_vals)))
    df = df.transpose()
    df = df.rename(columns={0: "adm2", 1: "adm2_name", 2: "adm1"})

    # Apply admin1 name map
    df["adm2"] = df["adm2"].astype(int)
    df["adm1"] = df["adm1"].astype(int)
    df["adm1_name"] = df["adm1"].map(admin1_name_map)

    # Add column for country
    df["adm0"] = admin0_name
    df["adm0_name"] = admin0_name

    # If all adm1 names are nan, use adm1 numeric as value
    if df["adm1_name"].isna().sum() == df["adm1_name"].size:
        df["adm1_name"] = df["adm1"].values.astype(str)

    df.set_index("adm2", inplace=True)

    # Deal with DC if in US
    if admin0_name == "US":
        df.loc[11001.0, "adm1_name"] = "District of Columbia"
    
    df.reset_index(inplace=True)
    return df


def read_lookup(geofile, country="US"):
    """Creates a dataframe relating geographic administration levels, 
    e.g. admin2 values in a given admin1 based on a lookup table.

    Parameters
    ----------
    geofile : string
        Location of lookup table
    country : string (default: 'US')
        Country name

    Returns
    -------
    df : Pandas DataFrame
        Dataframe with names and values for admin0, admin1, and admin2

    """

    df = pd.read_csv(geofile)

    # Make columns match the lookup tables created by graph
    df = df.rename(
        columns={"geoid": "adm2", "state_name": "adm1_name", "county_name": "adm2_name"}
    )

    # Add adm0 name
    df["adm0"] = country
    df["adm0_name"] = country

    # Create adm1 codes
    df["adm1"] = df["adm1_name"].astype("category").cat.codes
    df["adm2"] = df["adm2"].astype(int)
    df["adm1"] = df["adm1"].astype(int)

    # Drop FEMA region if it exists
    if "fema_region" in df.columns:
        df = df.drop(columns="fema_region")

    return df
