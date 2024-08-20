import pandas as pd
import datetime

import numpy as np
import json

from tqdm import tqdm

tqdm.pandas()

pd.options.mode.chained_assignment = None  # default='warn'


def read_and_preprocess(
    data: pd.DataFrame, data_distances: pd.DataFrame, numerus_fixus_list: list
) -> pd.DataFrame:
    """
    Read and preprocess the input data for further analysis.

    Args:
        data (pd.DataFrame): The raw data to be preprocessed.
        data_distances (pd.DataFrame): Data of distances from the Radboud.
        numerus_fixus_list (list): List of numerus fixus programmes

    Returns:
        pd.DataFrame: A preprocessed DataFrame ready for analysis.

    This function performs a series of data preprocessing steps on the input DataFrame:
    1. Removal of unnecessary columns.
    2. Filtering based on specific conditions.
    3. Grouping and counting data.
    4. Date transformation to week numbers.
    5. Determining the 'Herkomst' column based on nationality and EER status.
    6. Filtering data based on 'Ingangsdatum', 'Faculteit', and other criteria.
    7. Creating a 'is_numerus_fixus' column.
    8. Standardizing 'Examentype' values.
    9. Filtering data based on various conditions.
    10. Replacing less frequent 'Nationaliteit' values with 'Overig'.
    11. Adding city distances.
    12. Creating a 'Deadlineweek' column based on specific conditions.
    13. Additional data filtering based on year and exam type.
    14. Removal of the 'Sleutel' column.
    """

    # Remove redudant columns
    data = data.drop(labels=["Aantal studenten"], axis=1)

    # Filter english language and culture
    data = data[
        ~(
            (data["Croho groepeernaam"] == "B English Language and Culture")
            & (data.Collegejaar == 2021)
            & (data.Examentype != "Propedeuse Bachelor")
        )
    ]

    # Group the dataframe by collegejaar, weeknummer, and Sleutel
    grouped = data.groupby(["Collegejaar", "Sleutel"])

    # Create a new column in the original dataframe using the counts
    data["Sleutel_count"] = grouped["Sleutel"].transform("count")

    # Weeknummers

    def to_weeknummer(date):
        try:
            split_data = date.split("-")

            year = int(split_data[2])
            month = int(split_data[1])
            day = int(split_data[0])

            weeknummer = datetime.date(year, month, day).isocalendar()[1]

            return weeknummer
        except AttributeError:
            return np.nan

    data["Datum intrekking vooraanmelding"] = data["Datum intrekking vooraanmelding"].apply(
        to_weeknummer
    )

    data["Weeknummer"] = data["Datum Verzoek Inschr"].apply(to_weeknummer)

    def get_herkomst(nat, eer):
        if nat == "Nederlandse":
            return "NL"
        elif nat != "Nederlandse" and eer == "J":
            return "EER"
        else:
            return "Niet-EER"

    data["Herkomst"] = data.apply(lambda x: get_herkomst(x["Nationaliteit"], x["EER"]), axis=1)

    # Ingangsdatum
    data = data[
        data.Ingangsdatum.str.contains("01-09-") | data.Ingangsdatum.str.contains("01-10-")
    ]

    # Faculteit
    data.Faculteit = data.Faculteit.replace("RU", "FdM")

    # Create numerus fixus kolom
    data["is_numerus_fixus"] = (data["Croho groepeernaam"].isin(numerus_fixus_list)).astype(int)
    # ['B Geneeskunde', 'B Tandheelkunde', 'B Biology', 'B Artificial Intelligence', 'B Psychologie'])).astype(int)

    # Aanpassen 'Bachelor Eerstejaars' naar 'Bachelor'
    data["Examentype"] = data["Examentype"].replace("Propedeuse Bachelor", "Bachelor")

    # Filter data
    data = data[
        (data["Is eerstejaars croho opleiding"] == 1)
        & (data["Is hogerejaars"] == 0)
        & (data["BBC ontvangen"] == 0)
    ]

    data = data[data["Inschrijfstatus"].notna()]

    data = data.drop(
        [
            "Eerstejaars croho jaar",
            "Is eerstejaars croho opleiding",
            "Ingangsdatum",
            "BBC ontvangen",
            "Croho",
            "Is hogerejaars",
        ],
        axis=1,
    )

    data = data[data["Examentype"].isin(["Bachelor", "Master"])]

    # Nationaliteit
    # Count the occurrences of each value in the 'Nationaliteit' column
    nationaliteit_counts = data["Nationaliteit"].value_counts()

    # Get the values that occur less than 5 times
    values_to_change = nationaliteit_counts[nationaliteit_counts < 100].index

    # Replace the values with 'Overig'
    data["Nationaliteit"] = data["Nationaliteit"].replace(values_to_change, "Overig")

    # Steden toevoegen
    afstanden = data_distances

    data["Afstand"] = np.nan
    data["Afstand"] = data["Geverifieerd adres plaats"].map(
        afstanden.set_index("Geverifieerd adres plaats")["Afstand"]
    )

    # Define a function to apply the conditions
    def get_new_column(row):
        if row["Weeknummer"] == 17 and not row["Croho groepeernaam"] in numerus_fixus_list:
            # ["B Geneeskunde", "B Biology",
            #  "B Biomedische Wetenschappen", "B Psychologie",
            #  "B Tandheelkunde",
            #  "B Artificial Intelligence"]:
            return True
        else:
            return False

    # Apply the function to create a new column 'NewColumn' in the DataFrame
    data["Deadlineweek"] = data.apply(get_new_column, axis=1)

    # Alleen data na 2019 en zonder 2020/2023
    # data = data[(data.Collegejaar >= 2019) & (data.Collegejaar != 2020) & (data.Collegejaar != 2023)]
    # data = data[(data.Collegejaar != 2023)]

    # ALleen bachelor voor nu
    # data = data[data.Examentype == 'Bachelor']

    data = data.drop(["Sleutel"], axis=1)

    return data


def replace_string_to_float(data, row, key, i):
    new_data = row[key]
    if new_data == "":
        new_data = 0.0
    elif not isinstance(new_data, float) and not isinstance(new_data, int):
        new_data = np.float64(new_data.replace(".", "").replace(",", "."))

    data.at[i, key] = new_data


# Vooraanmeldingen toevoegen
def vooraanmeldingen_joinen(data: pd.DataFrame) -> pd.DataFrame:
    """
    Joins the main dataframe with the data with cumulative pre applications.

    Args:
        data (pd.DataFrame): The raw preapplication data to be joined and preprocessed.

    Returns:
        pd.DataFrame: A consolidated and preprocessed DataFrame containing preapplication information.

    This function performs the following tasks:
    1. Converts columns with comma-separated numbers to float64 data types.
    2. Groups the data by specific columns and sums non-numeric columns.
    3. Resets the index and selects relevant columns.
    4. Renames the 'Groepeernaam Croho' column to 'Croho groepeernaam'.
    """

    if "Weeknummer" not in data.columns:
        data = data.rename(columns={"Weeknummer rapportage": "Weeknummer"})

    data = data.rename(columns={"Type hoger onderwijs": "Examentype"})

    data = data[data["Hogerejaars"] == "Nee"]

    for i, row in data.iterrows():
        replace_string_to_float(data, row, "Gewogen vooraanmelders", i)
        replace_string_to_float(data, row, "Inschrijvingen", i)

    data = (
        data.groupby(
            [
                "Collegejaar",
                "Groepeernaam Croho",
                "Faculteit",
                "Examentype",
                "Herkomst",
                "Weeknummer",
            ]
        )
        .sum(numeric_only=False)
        .reset_index()
    )
    data = data[
        [
            "Weeknummer",
            "Collegejaar",
            "Faculteit",
            "Examentype",
            "Herkomst",
            "Groepeernaam Croho",
            "Gewogen vooraanmelders",
            "Ongewogen vooraanmelders",
            "Aantal aanmelders met 1 aanmelding",
            "Inschrijvingen",
        ]
    ]

    data = data.rename(columns={"Groepeernaam Croho": "Croho groepeernaam"})

    return data
