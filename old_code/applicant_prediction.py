import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier

from scripts.helper import get_weeks_list

from tqdm import tqdm

tqdm.pandas()

pd.options.mode.chained_assignment = None  # default='warn'


def make_applicant_prediction(
    clf, data: pd.DataFrame, week: int, year: int, max_year: int
) -> np.array:
    """
    This function makes the applicant predictions for every single pre-applicant. It predicts a value between
    0 and 1 with the chance that a pre-applicant actually results in an enrollment.

    Args:
        clf: The classification model used for prediction.
        data (pd.DataFrame): The input data for training and testing the model.
        week (int): The week for which predictions are made.
        year (int): The year for which predictions are made.
        max_year (int): The maximum year considered in the analysis.

    Returns:
        np.ndarray: An array of predicted probabilities for applicant statuses.
    """
    # Train/test split
    if year == max_year:
        train = data[(data.Collegejaar < year) & (data.Collegejaar >= 2016)]
        test = data[(data.Collegejaar == year)]
    else:
        train = data[
            (data.Collegejaar != year)
            & (data.Collegejaar >= 2016)
            & (data.Collegejaar != max_year)
        ]
        test = data[(data.Collegejaar == year)]

    # Weghalen geanuleerde inschrijvingen
    if int(week) < 39:
        train = train[
            (train["Datum intrekking vooraanmelding"].isna())
            | (
                (train["Datum intrekking vooraanmelding"] >= int(week))
                & (train["Datum intrekking vooraanmelding"] < 39)
            )
        ]
    elif int(week) > 38:
        train = train[
            (train["Datum intrekking vooraanmelding"].isna())
            | (
                (train["Datum intrekking vooraanmelding"] > int(week))
                | (train["Datum intrekking vooraanmelding"] < 39)
            )
        ]

    # Muteren inschrijfstatus
    status_map = {
        "Ingeschreven": 1,
        "Geannuleerd": 0,
        "Uitgeschreven": 1,
        "Verzoek tot inschrijving": 0,
        "Studie gestaakt": 0,
        "Aanmelding vervolgen": 0,
    }

    # use the map function to apply the mapping to the column
    train["Inschrijfstatus"] = train["Inschrijfstatus"].map(status_map)

    X_train = train.drop(["Inschrijfstatus"], axis=1)
    y_train = train.pop("Inschrijfstatus")

    X_test = test.drop(["Inschrijfstatus"], axis=1)
    y_test = test.pop("Inschrijfstatus")

    # Encode
    # Specify the numeric and categorical column names
    numeric_cols = [
        "Collegejaar",
        "Sleutel_count",
        "is_numerus_fixus",
        "Gewogen vooraanmelders",
        "Ongewogen vooraanmelders",
        "Aantal aanmelders met 1 aanmelding",
        "Inschrijvingen",
        "Afstand",
    ]  # Example numeric column names
    categorical_cols = [
        "Examentype",
        "Faculteit",
        "Croho groepeernaam",
        "Deadlineweek",
        "Herkomst",
        "Weeknummer",
        "Opleiding",
        "Type vooropleiding",
        "Nationaliteit",
        "EER",
        "Geslacht",
        "Plaats code eerste vooropleiding",
        "Studieadres postcode",
        "Studieadres land",
        "Geverifieerd adres plaats",
        "Geverifieerd adres land",
        "Geverifieerd adres postcode",
        "School code eerste vooropleiding",
        "School eerste vooropleiding",
        "Land code eerste vooropleiding",
    ]

    # Create transformers for numeric and categorical columns
    numeric_transformer = "passthrough"  # No transformation for numeric columns
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_cols),
            ("categorical", categorical_transformer, categorical_cols),
        ]
    )

    # Apply the preprocessing to the training and test data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Train model on the training data
    clf.fit(X_train, y_train)

    # Predict probability of all possible enrollment status of individual student
    voorspellingen = clf.predict_proba(X_test)[:, 1]

    predicties = np.zeros(len(voorspellingen))

    # Filter: Als de vooraanmelding al geannuleerd is dan wordt deze automatisch 0
    for i, (voorspelling, real) in enumerate(zip(voorspellingen, y_test)):
        if real == "Geannuleerd" and test["Datum intrekking vooraanmelding"].iloc[
            i
        ] in get_weeks_list(week):
            pred = 0
        else:
            pred = voorspelling

        predicties[i] = pred

    return predicties


def predict_preapplication(data: pd.DataFrame, year: int, max_year: int, week: int) -> np.array:
    """
    Filters the data for a given week and runs the make_applicant_prediction function for that week.

    Args:
        data (pd.DataFrame): The preapplication data used for predictions.
        year (int): The year for which predictions are made.
        max_year (int): The maximum year considered in the analysis.
        week (int): The specific week for which predictions are made.

    Returns:
        np.ndarray: An array of predicted probabilities for applicant statuses.
    """

    # Input
    data_train = data[data.Weeknummer.isin(get_weeks_list(week))]

    data_train = create_ratio(data_train)
    data_train = data_train.replace([np.inf, -np.inf], np.nan)

    # Output
    model = XGBClassifier(objective="binary:logistic", eval_metric="auc")

    predicties = make_applicant_prediction(model, data_train, week, year, max_year)

    return predicties


def create_ratio(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio features by dividing specific columns by 'Ongewogen vooraanmelders' and handling missing values.

    Args:
        data (pd.DataFrame): The input data for creating ratio features.

    Returns:
        pd.DataFrame: A DataFrame with ratio features added.
    """

    data = data.copy()

    def fix_zero(number):
        if number == 0:
            return np.nan
        else:
            return number

    data["Ongewogen vooraanmelders"] = data["Ongewogen vooraanmelders"].apply(fix_zero)

    data["Gewogen vooraanmelders"] = (
        data["Gewogen vooraanmelders"] / data["Ongewogen vooraanmelders"]
    )
    data["Gewogen vooraanmelders"] = data["Gewogen vooraanmelders"].fillna(1)

    data["Aantal aanmelders met 1 aanmelding"] = (
        data["Aantal aanmelders met 1 aanmelding"] / data["Ongewogen vooraanmelders"]
    )
    data["Aantal aanmelders met 1 aanmelding"] = data["Aantal aanmelders met 1 aanmelding"].fillna(
        1
    )

    data["Inschrijvingen"] = data["Inschrijvingen"] / data["Ongewogen vooraanmelders"]
    data["Inschrijvingen"] = data["Inschrijvingen"].fillna(1)

    return data


def multiply_by_disfactor(data, value, weeknummer, faculteit, examentype, numerus_fixus):
    """
    CURRENTLY NOT USED

       Multiply a value by a discount factor based on faculty, exam type, and week number. Mostly relevant when there is
       a certain constant over prediction in a certain week(s).

       Args:
           data (pd.DataFrame): The input data (not used in the calculation).
           value (float): The value to be multiplied.
           weeknummer (int): The week number.
           faculteit (str): The faculty name.
           examentype (str): The type of examination (e.g., 'Bachelor' or 'Master').
           numerus_fixus (bool): Indicates whether the program has a numerus fixus.

       Returns:
           float: The result of multiplying the value by the discount factor, or the original value.
    """

    # Alleen voor bachelor
    no_NF_rates = {
        "FdL": 0.75,
        "FdM": 0.67,
        "FdR": 0.81,
        "FFTR": 0.72,
        "FNWI": 0.68,
        "FSW": 0.67,
        "FMW": 1,
    }

    NF_rates = {"FNWI": 0.65, "FSW": 0.8, "FMW": 1}

    if examentype == "Bachelor":
        if not numerus_fixus and weeknummer == 17:
            return value * no_NF_rates[faculteit]
        elif numerus_fixus and weeknummer == 2:
            return value * NF_rates[faculteit]
        else:
            return value

    else:
        return value
