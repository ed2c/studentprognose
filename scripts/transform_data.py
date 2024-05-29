from scripts.helper import *

import pandas as pd
import numpy as np
from tqdm import tqdm
import collections
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

pd.options.mode.chained_assignment = None  # default='warn'


def transform(data: pd.DataFrame, target_year: int, last_week: int, old_method = True) -> pd.DataFrame:
    """
        Transforms the dataframe into a workable dataframe suitable for prediction. It groups the data and creates the
        cumulative sum of the pre-applications.

        Args:
            data (pd.DataFrame): The input data to be transformed.
            target_year (int): The target year for the transformation.
            last_week (int): The last week considered for the transformation.

        Returns:
            pd.DataFrame: The transformed data ready for analysis.
    """

    data = data[data.Collegejaar <= target_year]

    group_cols = ['Collegejaar', 'Faculteit', 'Herkomst', 'Examentype', 'Croho groepeernaam']

    # Create all weeks
    all_weeks = []
    all_weeks = all_weeks + [str(i) for i in range(39, 53)]
    all_weeks = all_weeks + [str(i) for i in range(1, 39)]

    # Create target weeks
    target_year_weeknummers = []
    if int(last_week) > 38:
        target_year_weeknummers = target_year_weeknummers + [str(i) for i in range(39, int(last_week) + 1)]
    elif int(last_week) < 39:
        target_year_weeknummers = target_year_weeknummers + [str(i) for i in range(39, 53)]
        target_year_weeknummers = target_year_weeknummers + [str(i) for i in range(1, int(last_week) + 1)]

    if old_method:
        data = data[group_cols + ['Inschrijvingen_predictie', 'Inschrijfstatus', 'Weeknummer']]
    else:
        data = data[group_cols + ['Weeknummer']]

    data['Weeknummer'] = data['Weeknummer'].astype(str)

    # Muteren inschrijfstatus
    status_map = {'Ingeschreven': 1,
                  'Geannuleerd': 0,
                  'Uitgeschreven': 1,
                  'Verzoek tot inschrijving': 0,
                  'Studie gestaakt': 0,
                  'Aanmelding vervolgen': 0
                  }

    # use the map function to apply the mapping to the column
    if old_method:
        data['Inschrijfstatus'] = data['Inschrijfstatus'].map(status_map)

    data = data.groupby(group_cols + ['Weeknummer']).sum(numeric_only=False).reset_index()

    # data = data[data.Collegejaar != 2023]

    def transform_data(input_data, target_col, weeknummers):
        data2 = input_data.reset_index().drop(['index', target_col], axis=1)
        # Pivot
        input_data = input_data.pivot(index=group_cols, columns='Weeknummer', values=target_col).reset_index()

        # Reorder columns
        input_data.columns = map(str, input_data.columns)
        colnames = group_cols + weeknummers
        input_data = input_data[colnames]

        input_data = input_data.fillna(0)
        input_data = input_data.melt(ignore_index=False, id_vars=group_cols, value_vars=weeknummers)

        input_data = input_data.rename(columns={'variable': 'Weeknummer', 'value': target_col})

        input_data = input_data.merge(data2, on=group_cols + ['Weeknummer'], how='left')

        input_data = input_data.fillna(0)

        input_data['Cumulative_sum_within_year'] = input_data.groupby(group_cols)[target_col].transform(
            pd.Series.cumsum)

        return input_data

    # Real data
    data_real = data[data.Collegejaar != target_year]
    data_real = transform_data(data_real, 'Inschrijfstatus', all_weeks)

    # Data predict
    data_predict = data[data.Collegejaar == target_year]
    data_predict = transform_data(data_predict, 'Inschrijvingen_predictie', target_year_weeknummers)

    data = pd.concat([data_real, data_predict])

    # data['Percentage'] = np.log1p(data['Percentage'])

    return data

def transform_data(data_input: pd.DataFrame, targ_col: str) -> pd.DataFrame:
    """
        Makes a certain pivot_wider where it transforms the data from long to wide

        Args:
            data_input (pd.DataFrame): The input data to be transformed.
            targ_col (str): The target column for pivoting.

        Returns:
            pd.DataFrame: The transformed data ready for analysis.
    """
    group_cols = ['Collegejaar', 'Faculteit', 'Herkomst', 'Examentype', 'Croho groepeernaam']

    # Only keep relevant columns
    data = data_input[group_cols + [targ_col, 'Weeknummer']]

    data = data.drop_duplicates()

    data = data.pivot(index=group_cols, columns='Weeknummer', values=targ_col).reset_index()

    # Reorder columns
    data.columns = map(str, data.columns)

    colnames = group_cols + get_all_weeks_valid(data.columns)

    data = data[colnames]

    # Fill na
    data = data.fillna(0)

    return data


def create_total_file(data: pd.DataFrame, vooraanmeldingen: pd.DataFrame, data_student_numbers: pd.DataFrame) -> pd.DataFrame:
    """
        Create a total file by combining data from the predictions, vooraanmeldingen and sioux

        Args:
            data (pd.DataFrame): The primary data source to be combined.
            vooraanmeldingen (pd.DataFrame): Additional data source to be merged.
            data_student_numbers (pd.DataFrame): The data of student numbers, used to merge with the primary data source.

        Returns:
            pd.DataFrame: The combined data ready for analysis.
        """

    '''
    sioux = pd.read_csv(
        '\\\\ru.nl\\wrkgrp\\TeamIR\\Man_info\\Student Analytics\\Prognosemodel RU\\Gijs\\data\\sioux.csv',
        sep=';')

    sioux = sioux[
        ['Type hoger onderwijs', 'Groepeernaam Croho', 'Herkomst', 'Faculteit', 'Collegejaar', 'Voorspelling',
         'Aantal_studenten', 'Current week number']]

    sioux = sioux.groupby(['Type hoger onderwijs', 'Groepeernaam Croho', 'Herkomst', 'Faculteit', 'Collegejaar',
                           'Current week number']).sum().reset_index()
    sioux = sioux.rename(columns={'Type hoger onderwijs': 'Examentype', 'Groepeernaam Croho': 'Croho groepeernaam',
                                  'Current week number': 'Weeknummer', 'Voorspelling': 'Voorspelling_Sioux'})

    sioux = sioux[
        ['Croho groepeernaam', 'Examentype', 'Faculteit', 'Collegejaar', 'Herkomst', 'Aantal_studenten', 'Weeknummer',
         'Voorspelling_Sioux']]
    # sioux['Weeknummer'] = sioux['Weeknummer'].astype(str)
    '''

    studentenaantallen = data_student_numbers

    data = data[['Croho groepeernaam', 'Collegejaar', 'Herkomst', 'Weeknummer', 'SARIMA_cumulative', 'SARIMA_individual', 'Voorspelde vooraanmelders']]

    total = data.merge(studentenaantallen, on=['Croho groepeernaam', 'Collegejaar', 'Herkomst'], how='left')

    total = total.merge(vooraanmeldingen, on=['Croho groepeernaam', 'Collegejaar', 'Herkomst', 'Weeknummer'],
                        how='left')

    return total

def replace_latest_data(data_latest: pd.DataFrame, data: pd.DataFrame, predict_year, predict_week):
    """
        Replace the weeks and years of the latest data with the forecasted data.
 
        Args:
            data_latest (pd.DataFrame): Latest data, from the totaal.xlsx file.
            data (pd.DataFrame): Forecasted data.
 
        Returns:
            pd.DataFrame: The new data_latest, with replaced data from the forecasted data
    """
 
    years_used = data["Collegejaar"].unique()
    weeks_used = data["Weeknummer"].unique()

    programmes_used = data["Croho groepeernaam"].unique()
    origins_used = data["Herkomst"].unique()
 
    # If there are no weeks/years of data in data_latest, then add data to data_latest
    if len(data_latest[data_latest["Collegejaar"].isin(years_used) & data_latest["Weeknummer"].isin(weeks_used)]) == 0:
        data_latest = pd.concat([data_latest, data], ignore_index=True)
    else:
        # Deletes the first week, because that has the actual prediction
        weeks_used = np.delete(weeks_used, 0)

        # Add the predicted pre-applicants to data_latest
        data_latest = data_latest.merge(data[data["Collegejaar"].isin(years_used) & data["Weeknummer"].isin(weeks_used)][['Croho groepeernaam', 'Collegejaar', 'Herkomst', 'Weeknummer', 'Voorspelde vooraanmelders']], on=['Croho groepeernaam', 'Collegejaar', 'Herkomst', 'Weeknummer'], how='left')
        data_latest = data_latest.drop(['Voorspelde vooraanmelders_x'], axis=1).rename(columns={"Voorspelde vooraanmelders_x": "Voorspelde vooraanmelders_y"})

        # Delete the data of the current predict_year and predict_week to make room for the newly predicted data, but keep in mind the programme and origin filtering
        data_latest = data_latest[~((data_latest["Collegejaar"] == predict_year) & (data_latest["Weeknummer"] == predict_week) &
                                    (data_latest["Croho groepeernaam"].isin(programmes_used)) & (data_latest["Herkomst"].isin(origins_used)))]
        data_latest = pd.concat([data_latest, data[(data["Collegejaar"] == predict_year) & (data["Weeknummer"] == predict_week)]], ignore_index=True)

    return data_latest

def replace_latest_data_old(data_latest: pd.DataFrame, data: pd.DataFrame):
    """
        Replace the weeks and years of the latest data with the forecasted data.

        Args:
            data_latest (pd.DataFrame): Latest data, from the totaal.xlsx file.
            data (pd.DataFrame): Forecasted data.

        Returns:
            pd.DataFrame: The new data_latest, with replaced data from the forecasted data
    """

    years_used = data["Collegejaar"].unique()
    weeks_used = data["Weeknummer"].unique()

    for year in years_used:
        for week in weeks_used:
            data_latest = data_latest[~((data_latest["Collegejaar"] == year) & (data_latest["Weeknummer"] == week))]

    data_latest = pd.concat([data_latest, data], ignore_index=True)

    return data_latest

def calculate_volume_predicted_data(data_first_years, data_second_years, predict_year, predict_week):
    data = data_first_years.merge(data_second_years, on=['Collegejaar', 'Faculteit', 'Examentype', 'Herkomst', 'Croho groepeernaam', 'Weeknummer'], how='left')

    data["SARIMA_cumulative"] = np.nan
    data["SARIMA_individual"] = np.nan
    data["Voorspelde vooraanmelders"] = np.nan

    for i, row in data.iterrows():
        if row["Collegejaar"] == predict_year and row["Weeknummer"] == predict_week:
            data.at[i, "SARIMA_cumulative"] = convert_nan_to_zero(row["SARIMA_cumulative_x"]) + convert_nan_to_zero(row["SARIMA_cumulative_y"])
            data.at[i, "SARIMA_individual"] = convert_nan_to_zero(row["SARIMA_individual_x"]) + convert_nan_to_zero(row["SARIMA_individual_y"])
        else:
            data.at[i, "Voorspelde vooraanmelders"] = convert_nan_to_zero(row["Voorspelde vooraanmelders_x"]) + convert_nan_to_zero(row["Voorspelde vooraanmelders_y"])

    return data[['Collegejaar', 'Faculteit', 'Examentype', 'Herkomst', 'Croho groepeernaam', 'Weeknummer', 'SARIMA_cumulative', 'SARIMA_individual', 'Voorspelde vooraanmelders']]

def sum_volume_data_cumulative(data_first_years, data_second_years):
    data = data_first_years.merge(data_second_years, on=['Weeknummer', 'Collegejaar', 'Faculteit', 'Examentype', 'Herkomst', 'Croho groepeernaam'], how='left')

    data["Gewogen vooraanmelders"] = np.nan
    data["Ongewogen vooraanmelders"] = np.nan
    data["Aantal aanmelders met 1 aanmelding"] = np.nan
    data["Inschrijvingen"] = np.nan

    for i, row in data.iterrows():
        data.at[i, "Gewogen vooraanmelders"] = convert_nan_to_zero(row["Gewogen vooraanmelders_x"]) + convert_nan_to_zero(row["Gewogen vooraanmelders_y"])
        data.at[i, "Ongewogen vooraanmelders"] = convert_nan_to_zero(row["Ongewogen vooraanmelders_x"]) + convert_nan_to_zero(row["Ongewogen vooraanmelders_y"])
        data.at[i, "Aantal aanmelders met 1 aanmelding"] = convert_nan_to_zero(row["Aantal aanmelders met 1 aanmelding_x"]) + convert_nan_to_zero(row["Aantal aanmelders met 1 aanmelding_y"])
        data.at[i, "Inschrijvingen"] = convert_nan_to_zero(row["Inschrijvingen_x"]) + convert_nan_to_zero(row["Inschrijvingen_y"])

    return data[['Weeknummer', 'Collegejaar', 'Faculteit',
                    'Examentype', 'Herkomst', 'Croho groepeernaam',
                    'Gewogen vooraanmelders', 'Ongewogen vooraanmelders',
                    'Aantal aanmelders met 1 aanmelding', 'Inschrijvingen']]