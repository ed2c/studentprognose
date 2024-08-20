import pandas as pd
import numpy as np

import statsmodels.api as sm

from xgboost import XGBRegressor

import gc
from numpy import linalg as LA
from tqdm import tqdm

from scripts.helper import get_weeks_list, increment_week
from scripts.transform_data import transform, transform_data
from scripts.applicant_prediction import predict_preapplication

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import joblib

import os
import time
import math

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

pd.options.mode.chained_assignment = None  # default='warn'

numerus_fixus_list = []

# Sarima voorspelling


def predict_with_sarima(
    data: pd.DataFrame,
    data_exog: pd.DataFrame,
    opl: str,
    herkomst: str,
    weeknummer: int,
    jaar: int,
    max_year: int,
) -> float:
    """
    Predicts a value using SARIMA (Seasonal Autoregressive Integrated Moving Average) modeling.

    Args:
        data (pd.DataFrame): Main data containing time series.
        data_exog (pd.DataFrame): Exogenous data used for modeling.
        opl (str): Study program.
        herkomst (str): Origin information (NL/EER/niet-EER).
        weeknummer (int): Week number for which prediction is made.
        jaar (int): Year for which prediction is made.
        max_year (int): Maximum year in the dataframe.

    Returns:
        float: Predicted amount of students for a specific combination of a opleiding/herkomst, in a certain year.
    """
    gc.collect()

    global numerus_fixus_list

    def filter_data(
        data: pd.DataFrame, opl: str, herkomst: str, jaar: int, max_year: int
    ) -> pd.DataFrame:
        """
        Filters a given dataframe based on opleiding, herkomst and jaar. Returns the filtered dataframe

        Args:
            data (pd.DataFrame): Main data to be filtered.
            opl (str): Study program code.
            herkomst (str): Origin information (NL/EER/niet-EER).
            jaar (int): Year for which filtering is applied.
            max_year (int): Maximum year in the dataframe.

        Returns:
            pd.DataFrame: Filtered data.
        """
        data = data[data.Herkomst == herkomst]

        if jaar != max_year:
            data = data[data.Collegejaar <= jaar]

        data = data[data["Croho groepeernaam"] == opl]

        return data

    # Filter both datasets
    data_exog = filter_data(data_exog, opl, herkomst, jaar, max_year)
    data = filter_data(data, opl, herkomst, jaar, max_year)

    def deadline_week(weeknummer, croho, examentype):
        """
        Determines if a week corresponds to a deadline week for a specific study program.
        This is added as exogenous variabele. Master does not seem to have a strong deadline week.

        Args:
            weeknummer (int): Week number.
            croho (str): Croho group name.
            examentype (str): Type of exam.

        Returns:
            int: 1 if it's a deadline week, 0 otherwise.
        """

        # numerus_fixus = ["B Geneeskunde", "B Biology", "B Biomedische Wetenschappen",
        #                  "B Psychologie", "B Tandheelkunde", "B Artificial Intelligence"]

        if weeknummer in [16, 17] and examentype == "Bachelor" and croho not in numerus_fixus_list:
            return 1
        elif weeknummer in [1, 2] and examentype == "Bachelor" and croho in numerus_fixus_list:
            return 1
        else:
            return 0

    # Apply the 'deadline_week' function on the dataset
    data_exog["Deadline"] = data_exog.apply(
        lambda x: deadline_week(x.Weeknummer, x["Croho groepeernaam"], x.Examentype),
        axis=1,
    )

    try:
        data_exog = transform_data(data_exog, "Deadline")

        if weeknummer == 38:
            ts_data = data.loc[:, "39":"38"].values.flatten()
            try:
                return ts_data[-1]
            except IndexError:
                return np.nan

        if int(weeknummer) > 38:
            pred_len = 38 + 52 - int(weeknummer)
        else:
            pred_len = 38 - int(weeknummer)

        def create_time_series(data: pd.DataFrame, pred_len: int) -> np.array:
            """
            Create a time series data array from a DataFrame for a given prediction length.

            Args:
                data (pd.DataFrame): The input DataFrame containing time series data.
                pred_len (int): The length of the time series to be created.

            Returns:
                np.ndarray: A NumPy array containing the time series data.
            """

            ts_data = data.loc[:, "39":"38"].values.flatten()
            ts_data = ts_data[:-pred_len]

            return np.array(ts_data)

        def create_exogenous(data: pd.DataFrame, pred_len: int) -> np.array:
            """
            Create an exogenous time series data array from a DataFrame for a given prediction length.

            Args:
                data (pd.DataFrame): The input DataFrame containing time series data.
                pred_len (int): The length of the time series to be created.

            Returns:
                np.ndarray: A NumPy array containing the time series data.
            """

            exg_data = data.loc[:, "39":"38"].values.flatten()
            exg_data_train = exg_data[:-pred_len]
            exg_data_test = exg_data[-pred_len:]

            return np.array(exg_data_train), np.array(exg_data_test)

        ts_data = create_time_series(data, pred_len)

        exogenous_train_1, exg_data_test_1 = create_exogenous(data_exog, pred_len)
        # exogenous_train_2, exg_data_test_2 = create_exogenous(data_exog, 'Gewogen vooraanmelders', pred_len)
        # exogenous_train_3, exg_data_test_3 = create_exogenous(data_exog, 'Ongewogen vooraanmelders', pred_len)
        # exogenous_train_4, exg_data_test_4 = create_exogenous(data_exog, 'Aantal aanmelders met 1 aanmelding', pred_len)
        # exogenous_train_5, exg_data_test_5 = create_exogenous(data_exog, 'Inschrijvingen', pred_len)

        # exogenous_train = np.column_stack((exogenous_train_1, exogenous_train_5))
        # exg_data_test = np.column_stack((exg_data_test_1,  exg_data_test_5))
        try:
            # Create SARIMA
            weeknummers = [17, 18, 19, 20, 21]
            if opl.startswith("B") and weeknummer in weeknummers:
                # This model seems to work better for bachelor programmes close to the deadline
                model = sm.tsa.statespace.SARIMAX(
                    ts_data,
                    order=(1, 0, 1),
                    seasonal_order=(1, 1, 1, 52),
                    exog=exogenous_train_1,
                )
            else:
                # For the other time series I used these settings for SARIMA
                model = sm.tsa.statespace.SARIMAX(
                    ts_data,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 0, 52),
                    exog=exogenous_train_1,
                )
            results = model.fit(disp=0)
            pred = results.forecast(steps=pred_len, exog=exg_data_test_1)

            # Only the last prediction is relevant
            return pred[-1]
        except (LA.LinAlgError, IndexError, ValueError):
            print("Error")
            pass
    except KeyError:
        print("Error")
        pass


def predict_with_xgboost(data, year, opl, examentype, herkomst, totaal):
    global numerus_fixus_list

    try:
        # Train/test split
        # NF = ["B Geneeskunde", "B Biology", "B Biomedische Wetenschappen", "B Psychologie", "B Tandheelkunde",
        #       "B Artificial Intelligence"]
        if opl not in numerus_fixus_list:
            train = data[
                (data.Collegejaar < year)
                & (data.Examentype == examentype)
                & (~data["Croho groepeernaam"].isin(numerus_fixus_list))
            ]
        elif opl in numerus_fixus_list:
            train = data[(data.Collegejaar < year) & (data["Croho groepeernaam"] == opl)]

        test = data[
            (data.Collegejaar == year)
            & (data["Croho groepeernaam"] == opl)
            & (data.Herkomst == herkomst)
        ]

        train = train.merge(
            totaal[["Croho groepeernaam", "Collegejaar", "Herkomst", "Aantal_studenten"]],
            on=["Croho groepeernaam", "Collegejaar", "Herkomst"],
        )
        train = train.drop_duplicates()

        X_train = train.drop(["Aantal_studenten"], axis=1)
        y_train = train.pop("Aantal_studenten")

        # X_test = test.drop(['Aantal_studenten'], axis=1)

        # Encode
        # Specify the numeric and categorical column names
        numeric_cols = ["Collegejaar"] + [str(x) for x in get_weeks_list(38)]
        categorical_cols = ["Examentype", "Faculteit", "Croho groepeernaam", "Herkomst"]

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
        test = preprocessor.transform(test)
        # Model
        OPTIONS = {
            # "objective": "regression_l1"
            "min_data_in_leaf": 10,
            "num_leaves": 10,  # 10 if linear_tree = True
            "max_depth": 50,
            "linear_tree": True,
        }

        model = XGBRegressor(learning_rate=0.25)

        # model = lgb.sklearn.LGBMRegressor(**OPTIONS)

        model.fit(X_train, y_train)

        predictie = model.predict(test)

        return int(round(predictie[0], 0))

    except ValueError:
        return np.nan


def predict_sarima(data, jaar, weeknummer, opl, examentype, herkomst, faculty, totaal):
    gc.collect()

    data = data.drop_duplicates()

    data = data[data.Collegejaar >= 2016]

    full_data = transform_data(data, "ts")

    data = full_data.copy()

    data = data[data.Herkomst == herkomst]

    data = data[data.Collegejaar <= jaar]

    data = data[data["Croho groepeernaam"] == opl]

    if int(weeknummer) > 38:
        pred_len = 38 + 52 - int(weeknummer)
    else:
        pred_len = 38 - int(weeknummer)

    # Week 39 to 0
    data["39"] = 0
    full_data["39"] = 0

    def create_time_series(data: pd.DataFrame, pred_len: int) -> np.array:
        """
        Create a time series data array from a DataFrame for a given prediction length.

        Args:
            data (pd.DataFrame): The input DataFrame containing time series data.
            pred_len (int): The length of the time series to be created.

        Returns:
            np.ndarray: A NumPy array containing the time series data.
        """

        ts_data = data.loc[:, "39":"38"].values.flatten()
        ts_data = ts_data[:-pred_len]

        return np.array(ts_data)

    ts_data = create_time_series(data, pred_len)
    if weeknummer == 38:
        prediction = predict_with_xgboost(full_data, jaar, opl, examentype, herkomst, totaal)
        return prediction

    try:
        model = sm.tsa.statespace.SARIMAX(ts_data, order=(1, 0, 1), seasonal_order=(1, 1, 1, 52))
        results = model.fit(disp=0)

        # model = ExponentialSmoothing(ts_data, seasonal='add', seasonal_periods=52)
        # results = model.fit(smoothing_level = 0.5, smoothing_trend = 0.1)

        pred = results.forecast(steps=pred_len)

        index = str(increment_week(weeknummer))

        # print("Index:", index)
        # print("Prediction:", pred)

        full_data.loc[
            (full_data["Collegejaar"] == jaar)
            & (full_data["Croho groepeernaam"] == opl)
            & (full_data["Herkomst"] == herkomst),
            index:"38",
        ] = pred

        # print(full_data)

        prediction = predict_with_xgboost(full_data, jaar, opl, examentype, herkomst, totaal)
        return prediction, (jaar, opl, herkomst, faculty, examentype, pred)
    except (LA.LinAlgError, IndexError, ValueError):
        return np.nan, (None, None, None, None, None, np.nan)


def make_full_week_prediction(
    data: pd.DataFrame,
    year: int,
    max_year: int,
    week: int,
    last_week: int,
    vooraanmeldingen: pd.DataFrame,
    total: pd.DataFrame,
    only_individual: bool,
    local_numerus_fixus_list: list,
) -> pd.DataFrame:
    """
    Make predictions for a full week (so all herkomst/opleiding combinations) based on preapplication data and SARIMA.

    Args:
        data (pd.DataFrame): The main data containing information for predictions.
        year (int): The year for which predictions are made.
        max_year (int): The maximum year considered in the analysis.
        week (int): The specific week for which predictions are made.
        last_week (int): The last week considered in the analysis.
        vooraanmeldingen (pd.DataFrame): Exogenous data containing preapplication information.
        numerus_fixus_list (list): List of numerus fixus programmes.

    Returns:
        pd.DataFrame: A DataFrame containing predictions for the specified week.
    """

    global numerus_fixus_list
    numerus_fixus_list = local_numerus_fixus_list

    # total = total[['Croho groepeernaam', 'Collegejaar', 'Herkomst', 'Weeknummer', 'Aantal_studenten']].drop_duplicates()

    # First predict for every pre-application (row) the chance that it will result in a real application
    predicties = predict_preapplication(data, year, max_year, week)
    data.loc[
        (data.Collegejaar == year) & (data.Weeknummer.isin(get_weeks_list(week))),
        "Inschrijvingen_predictie",
    ] = predicties

    data = transform(data, year, last_week)

    # Transform data so that is grouped and cumulative for the different opleiding/herkomst combinations
    data_totaal_cumulatief = transform_data(data, targ_col="Cumulative_sum_within_year")

    # Create the exogenous variables by merging the data with the vooraanmeldingen
    data.Weeknummer = data.Weeknummer.astype(int)
    data_exog = data.merge(
        vooraanmeldingen,
        on=[
            "Croho groepeernaam",
            "Collegejaar",
            "Examentype",
            "Faculteit",
            "Weeknummer",
            "Herkomst",
        ],
        how="left",
    )

    # We don't want to predict all years/weeks so this filters only the relevant weeks
    predict_df = vooraanmeldingen[
        (vooraanmeldingen.Collegejaar == year) & (vooraanmeldingen.Weeknummer == week)
    ]

    # print(predict_df)

    # total = total[['Croho groepeernaam', 'Collegejaar', 'Herkomst', 'Weeknummer', 'Aantal_studenten']]

    vooraanmeldingen = vooraanmeldingen.merge(
        total, on=["Croho groepeernaam", "Collegejaar", "Herkomst"], how="left"
    )
    vooraanmeldingen["ts"] = (
        vooraanmeldingen["Gewogen vooraanmelders"] + vooraanmeldingen["Inschrijvingen"]
    )

    vooraanmeldingen = vooraanmeldingen.drop_duplicates()

    vooraanmeldingen["Faculteit"] = vooraanmeldingen["Faculteit"].replace(
        {
            "SOW": "FSW",
            "LET": "FdL",
            "FTR": "FFTR",
            "NWI": "FNWI",
            "MAN": "FdM",
            "JUR": "FdR",
            "MED": "FMW",
            "RU": "FdM",
        }
    )
    """
    # Apply the predict_with_sarima function on the prediction data set.
    predict_df['SARIMA_cumulative'] = predict_df.progress_apply(
        lambda x: predict_sarima(vooraanmeldingen, x.Collegejaar, x.Weeknummer,
                                 x['Croho groepeernaam'], x.Examentype, x.Herkomst, total), axis=1)

    # Apply the predict_with_sarima function on the prediction data set.
    predict_df['SARIMA_individual'] = predict_df.progress_apply(
        lambda x: predict_with_sarima(data_totaal_cumulatief, data_exog, x['Croho groepeernaam'], x.Herkomst,
                                      x.Weeknummer, x.Collegejaar, max_year), axis=1)

    """

    # Define your custom function
    def sarima(row, data_cumu, data_exog, vooraanmeldingen, total, max_year):
        # print(vooraanmeldingen)
        print(
            f"Voorspelling maken voor {row['Croho groepeernaam']}, {row['Examentype']}, {row['Herkomst']}, jaar: {row['Collegejaar']}, week: {row['Weeknummer']}"
        )
        sarima_ind = predict_with_sarima(
            data_cumu,
            data_exog,
            row["Croho groepeernaam"],
            row["Herkomst"],
            row["Weeknummer"],
            row["Collegejaar"],
            max_year,
        )

        sarima_cumu, predicted_vooraanmelding = predict_sarima(
            vooraanmeldingen,
            row["Collegejaar"],
            row["Weeknummer"],
            row["Croho groepeernaam"],
            row["Examentype"],
            row["Herkomst"],
            row["Faculteit"],
            total,
        )

        return sarima_ind, sarima_cumu, predicted_vooraanmelding

    # Split the DataFrame into smaller chunks for parallel processing
    nr_CPU_cores = os.cpu_count()
    chunk_size = math.ceil(
        len(predict_df) / nr_CPU_cores
    )  # Make as much chunks as you have CPU cores

    chunks = [predict_df[i : i + chunk_size] for i in range(0, len(predict_df), chunk_size)]

    start = time.time()
    print("Starting parallel processes")

    # Use joblib.Parallel to parallelize the operation
    results = joblib.Parallel(n_jobs=nr_CPU_cores)(
        joblib.delayed(sarima)(
            row, data_totaal_cumulatief, data_exog, vooraanmeldingen, total, max_year
        )
        for chunk in chunks
        for _, row in chunk.iterrows()
    )

    end = time.time()
    print("End of parallel processes, time it took:", end - start)

    # Combine the results into a list
    SARIMA_individual = [result[0] for result in results]
    SARIMA_cumulative = [result[1] for result in results]
    predicted_vooraanmeldingen = [result[2] for result in results]

    # Convert the list to a Pandas DataFrame
    predict_df["SARIMA_individual"] = SARIMA_individual
    predict_df["SARIMA_cumulative"] = SARIMA_cumulative
    predict_df["Voorspelde vooraanmelders"] = np.nan

    predict_df = predict_df[
        [
            "Collegejaar",
            "Faculteit",
            "Examentype",
            "Herkomst",
            "Croho groepeernaam",
            "Weeknummer",
            "SARIMA_cumulative",
            "SARIMA_individual",
            "Voorspelde vooraanmelders",
        ]
    ]

    for (
        year,
        programme,
        herkomst,
        faculty,
        examtype,
        prediction,
    ) in predicted_vooraanmeldingen:
        current_week = increment_week(week)
        for pred in prediction:

            dict = {
                "Collegejaar": [year],
                "Faculteit": [faculty],
                "Examentype": [examtype],
                "Herkomst": [herkomst],
                "Croho groepeernaam": [programme],
                "Weeknummer": [current_week],
                "SARIMA_cumulative": [np.nan],
                "SARIMA_individual": [np.nan],
                "Voorspelde vooraanmelders": [pred],
            }
            df = pd.DataFrame(dict)

            predict_df = pd.concat([predict_df, df], ignore_index=True)

            current_week = increment_week(current_week)

    return predict_df
