""" Functions for loading the wmpl trajectory summary files. """

import os

import pandas as pd


def _set_data_types(dataframe: pd.DataFrame) -> None:
    """
    Sets the data types and index column in a DataFrame containing meteor trajectory
     data. The input dataframe must be in verbose column name format e.g.
     "Beginning (UTC Time)".

    Arguments: 
        dataframe: [Pandas df] The meteor trajectory dataframe to set the data types for.
        
        
    Return: None.
    
    """

    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

    dataframe["Beginning (UTC Time)"] = pd.to_datetime(
        dataframe["Beginning (UTC Time)"], format=DATETIME_FORMAT
    )
    dataframe["IAU (code)"] = dataframe[
        "IAU (code)"].astype("string")
    dataframe["IAU (No)"] = (
        dataframe["IAU (No)"].fillna(-1).astype("int64")
    )
    dataframe["Beg in (FOV)"] = dataframe[
        "Beg in (FOV)"].map(
        {"True": True, "False": False}
    )
    dataframe["Beg in (FOV)"] = dataframe[
        "Beg in (FOV)"].astype("bool")
    dataframe["End in (FOV)"] = dataframe[
        "End in (FOV)"].map(
        {"True": True, "False": False}
    )
    dataframe["End in (FOV)"] = dataframe[
        "End in (FOV)"].astype("bool")
    dataframe["Participating (stations)"] = dataframe[
        "Participating (stations)"
    ].astype("string")
    dataframe["Participating (stations)"] = dataframe[
        "Participating (stations)"
    ].apply(lambda x: x.strip().split(","))

    dataframe.set_index("Unique trajectory (identifier)", inplace=True)



def loadTrajectorySummary(dir_path, file_name):
    """ Loads the meteor trajectory summary file into a pandas DataFrame.
    
    Arguments:
        dir_path: [str] Path to the directory containing the trajectory summary file.
        file_name: [str] Name of the trajectory summary file.

    Returns:
        meteor_trajectory_df: [pandas.DataFrame] DataFrame containing the meteor trajectory data.
    """

    file_path = os.path.join(dir_path, file_name)

    meteor_trajectory_df = pd.read_csv(
            file_path,
            engine="python",
            sep=";",
            skipinitialspace=True,
            skiprows=[0, 5, 6],
            header=[0, 1],
            na_values=["nan", "...", "None"]
        )

    def extract_header(text: str) -> str:
        return " ".join(text.replace("#", "").split())

    meteor_trajectory_df.columns = meteor_trajectory_df.columns.map(
        lambda h: extract_header(h[0]) + (
            f" ({extract_header(h[1])})" if "Unnamed" not in h[1] else "")
    )

    # Section to create unique sigma column names
    new_columns = list(meteor_trajectory_df.columns)
    for i, col in enumerate(new_columns):
        if col.startswith('+/-'):
            # Get the name of the previous column (e.g., "RAgeo (deg)")
            parameter_col_name = new_columns[i-1]
            
            # Extract the first word (e.g., "RAgeo")
            parameter_name = parameter_col_name.split(' ')[0]
            
            # Create the new unique name and update it in the list
            new_columns[i] = f"{parameter_name}_sigma"
            
    # Assign the list of unique column names back to the DataFrame
    meteor_trajectory_df.columns = new_columns

    _set_data_types(meteor_trajectory_df)

    return meteor_trajectory_df


def loadTrajectorySummaryFast(dir_path_traj_summary, traj_summary_file_name, quick_file_name):
    """ Loads the meteor trajectory data from the CSV file into a pandas DataFrame.

    Arguments:
        dir_path_traj_summary: [str] Path to the directory containing the trajectory summary file.
        traj_summary_file_name: [str] Name of the trajectory summary file.
        quick_file_name: [str] Name of the pickle file for quick loading of the trajectory data.

    Returns:
        data: [pandas.DataFrame] DataFrame containing the meteor trajectory data.
    """

        # If the quick file is available, load the data from it
    quick_path = os.path.join(dir_path_traj_summary, quick_file_name)
    if os.path.exists(quick_path):
        print("Loading the quick load file...")
        data = pd.read_pickle(quick_path)
        

    # Otherwise, load the data from the CSV file
    else:

        print("Loading the trajectory CSV file...")
        data = loadTrajectorySummary(dir_path_traj_summary, traj_summary_file_name)
        print(data)

        # Save the data in the quick load format
        print("Saving the quick load file...")
        data.to_pickle(quick_path)


    return data