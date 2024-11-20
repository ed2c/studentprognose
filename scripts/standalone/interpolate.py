import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os as os

path_cumulative = "//ru.nl/wrkgrp/TeamIR/Man_info/Student Analytics/Prognosemodel RU/Syntax/Python/studentprognose/data/input/vooraanmeldingen_cumulatief.csv"
data_cumulative = (
    pd.read_csv(path_cumulative, sep=";", low_memory=True)
    if (path_cumulative != "" and os.path.exists(path_cumulative))
    else None
)


def convert_to_float(value):
    if isinstance(value, str):
        # Remove thousand separator (dot)
        value = value.replace(".", "")
        # Replace decimal separator (comma) with a dot
        value = value.replace(",", ".")
    return float(value)


# Function to convert other numeric columns to float
def convert_other_columns(value):
    if isinstance(value, str):
        # Remove thousand separator (dot)
        value = value.replace(".", "")
    return float(value)


# Apply the conversion function to the "Gewogen vooraanmelders" column
data_cumulative["Gewogen vooraanmelders"] = data_cumulative["Gewogen vooraanmelders"].apply(
    convert_to_float
)


def convert_other_columns(value):
    if isinstance(value, str):
        # Remove thousand separator (dot) and replace decimal separator (comma) with a dot
        value = value.replace(".", "").replace(",", ".")
    return float(value)


for col in ["Ongewogen vooraanmelders", "Aantal aanmelders met 1 aanmelding", "Inschrijvingen"]:
    data_cumulative[col] = data_cumulative[col].apply(convert_other_columns)


def interpolate(data):
    """
    Automatically fills values for weeks 32 and 33 by interpolating between weeks 31 and 34,
    but only for the year 2024. Returns the updated dataframe with the interpolated values.
    """

    # Step 1: Filter the data for the relevant year, weeks, and origin groups
    filtered_data = data[
        (data["Herinschrijving"] == "Nee")
        & (data["Hogerejaars"] == "Nee")
        & (data.Collegejaar == 2024)
        & (data.Weeknummer.isin([31, 34]))
        & (data.Herkomst.isin(["Niet-EER", "EER"]))
    ]

    # Step 2: Define the interpolation function for a single group (croho, herkomst)
    def interpolate_single_week(croho, herkomst, target_value):
        group_data = filtered_data[
            (filtered_data["Groepeernaam Croho"] == croho) & (filtered_data.Herkomst == herkomst)
        ]

        value_wk31 = group_data[group_data.Weeknummer == 31][target_value]
        value_wk34 = group_data[group_data.Weeknummer == 34][target_value]

        if not value_wk31.empty and not value_wk34.empty:
            # Convert values to float after ensuring the column was preprocessed
            wk31_value = convert_to_float(value_wk31.iloc[0])  # Ensure this is a float
            wk34_value = convert_to_float(value_wk34.iloc[0])  # Ensure this is a float

            # Interpolating for weeks 32 and 33 using a linear function
            interpolator = interp1d([31, 34], [wk31_value, wk34_value], kind="linear")
            return interpolator([32, 33])

        return None, None  # Return None if data is missing

    # Step 3: Initialize a dictionary to collect interpolated values
    updates = {
        "Collegejaar": [],
        "Weeknummer": [],
        "Groepeernaam Croho": [],
        "Herkomst": [],
        "target_value": [],
        "value": [],
    }

    # Step 4: Iterate over unique (croho, herkomst) combinations and interpolate
    croho_herkomst_groups = filtered_data[["Groepeernaam Croho", "Herkomst"]].drop_duplicates()

    for _, group in croho_herkomst_groups.iterrows():
        croho = group["Groepeernaam Croho"]
        herkomst = group["Herkomst"]

        for target_value in [
            "Gewogen vooraanmelders",
            "Ongewogen vooraanmelders",
            "Aantal aanmelders met 1 aanmelding",
            "Inschrijvingen",
        ]:
            wk32_value, wk33_value = interpolate_single_week(croho, herkomst, target_value)

            if wk32_value is not None and wk33_value is not None:
                # Collect values for week 32
                updates["Collegejaar"].append(2024)
                updates["Weeknummer"].append(32)
                updates["Groepeernaam Croho"].append(croho)
                updates["Herkomst"].append(herkomst)
                updates["target_value"].append(target_value)
                updates["value"].append(wk32_value)

                # Collect values for week 33
                updates["Collegejaar"].append(2024)
                updates["Weeknummer"].append(33)
                updates["Groepeernaam Croho"].append(croho)
                updates["Herkomst"].append(herkomst)
                updates["target_value"].append(target_value)
                updates["value"].append(wk33_value)

    # Step 5: Convert the updates dictionary into a DataFrame and pivot it
    updates_df = pd.DataFrame(updates)
    updates_pivot = updates_df.pivot_table(
        index=["Collegejaar", "Weeknummer", "Groepeernaam Croho", "Herkomst"],
        columns="target_value",
        values="value",
    ).reset_index()

    # Step 6: Merge the interpolated values back into the original dataset
    data_updated = data.merge(
        updates_pivot,
        on=["Collegejaar", "Weeknummer", "Groepeernaam Croho", "Herkomst"],
        how="left",
        suffixes=("", "_interpolated"),
    )

    # Step 7: Replace original values with interpolated ones where available
    for col in [
        "Gewogen vooraanmelders",
        "Ongewogen vooraanmelders",
        "Aantal aanmelders met 1 aanmelding",
        "Inschrijvingen",
    ]:
        data_updated[col] = data_updated[f"{col}_interpolated"].combine_first(data_updated[col])

    # Step 8: Drop the temporary interpolated columns
    data_updated.drop(
        columns=[
            f"{col}_interpolated"
            for col in [
                "Gewogen vooraanmelders",
                "Ongewogen vooraanmelders",
                "Aantal aanmelders met 1 aanmelding",
                "Inschrijvingen",
            ]
        ],
        inplace=True,
    )

    return data_updated


data_cumulative_met_interpolate = interpolate(data_cumulative)

# Round 'Gewogen vooraanmelders' to two decimal places
data_cumulative_met_interpolate["Gewogen vooraanmelders"] = data_cumulative_met_interpolate[
    "Gewogen vooraanmelders"
].round(2)

# Round other columns to whole numbers
columns_to_round = [
    "Ongewogen vooraanmelders",
    "Aantal aanmelders met 1 aanmelding",
    "Inschrijvingen",
]

for col in columns_to_round:
    # Round values only if they are finite
    data_cumulative_met_interpolate[col] = data_cumulative_met_interpolate[col].where(
        data_cumulative_met_interpolate[col].notna(), np.nan
    )  # Keep NaN as it is
    data_cumulative_met_interpolate[col] = np.round(data_cumulative_met_interpolate[col]).astype(
        "Int64"
    )  # Use 'Int64' to keep NaN


data_cumulative_met_interpolate.to_csv(
    "//ru.nl/wrkgrp/TeamIR/Man_info/Student Analytics/Prognosemodel RU/Syntax/Python/studentprognose/data/input/vooraanmeldingen_cumulatief.csv",
    sep=";",
    index=False,
)


# Filter the DataFrame for the year 2024
# data_2024 = data_cumulative_met_interpolate[data_cumulative_met_interpolate['Collegejaar'] == 2024]
