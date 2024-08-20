import numpy as np
import pandas as pd

### COLUMN PREDICTION ENSEMBLE ###

# IF [Croho groepeernaam] IN ('B Geneeskunde', 'B Biomedische Wetenschappen', 'B Tandheelkunde') THEN
# [SARIMA new]

# ELSEIF [Weeknummer] IN (17,18,19,20,21,22,23) AND [Examentype] == 'Master' THEN
# ([SARIMA old] * 0.20 + [SARIMA new] * 0.80)

# ELSEIF [Weeknummer] IN (30,31,32,33,34) THEN
# ([SARIMA old] * 0.60 + [SARIMA new] * 0.40)

# ELSEIF  [Weeknummer] IN (35,36,37) THEN
# ([SARIMA old] * 0.7 + [SARIMA new] * 0.3)

# ELSEIF  [Weeknummer] IN (38) THEN
# ([SARIMA old])

# ELSE
# ([SARIMA old] * 0.5 + [SARIMA new] * 0.5)
# END


def get_normal_ensemble(row):
    ensemble_prediction = None
    if row["Croho groepeernaam"] in [
        "B Geneeskunde",
        "B Biomedische Wetenschappen",
        "B Tandheelkunde",
    ]:
        ensemble_prediction = row["SARIMA_cumulative"]

    elif row["Weeknummer"] in range(17, 23 + 1) and row["Examentype"] == "Master":
        ensemble_prediction = row["SARIMA_individual"] * 0.2 + row["SARIMA_cumulative"] * 0.8

    elif row["Weeknummer"] in range(30, 34 + 1):
        ensemble_prediction = row["SARIMA_individual"] * 0.6 + row["SARIMA_cumulative"] * 0.4

    elif row["Weeknummer"] in range(35, 37 + 1):
        ensemble_prediction = row["SARIMA_individual"] * 0.7 + row["SARIMA_cumulative"] * 0.3

    elif row["Weeknummer"] == 38:
        ensemble_prediction = row["SARIMA_individual"]

    else:
        ensemble_prediction = row["SARIMA_individual"] * 0.5 + row["SARIMA_cumulative"] * 0.5

    return ensemble_prediction


def create_ensemble_columns(data, weighted_data):
    data = data.sort_values(by=["Croho groepeernaam", "Herkomst", "Collegejaar", "Weeknummer"])
    data = data.reset_index(drop=True)

    data["Ensemble_prediction"] = np.nan
    data["Weighted_ensemble_predicition"] = -1.0
    for index, row in data.iterrows():
        normal_ensemble = get_normal_ensemble(row)

        data.at[index, "Ensemble_prediction"] = normal_ensemble

        temp_weighted_data = weighted_data[weighted_data["Programme"] == row["Croho groepeernaam"]]
        temp_weighted_data = temp_weighted_data[temp_weighted_data["Herkomst"] == row["Herkomst"]]
        if len(temp_weighted_data) > 0:
            temp_weighted_data = temp_weighted_data.iloc[0]

            if temp_weighted_data["Average_ensemble_prediction"] != 1:
                weighted_ensemble = (
                    row["SARIMA_cumulative"] * temp_weighted_data["SARIMA_cumulative"]
                    + row["SARIMA_individual"] * temp_weighted_data["SARIMA_individual"]
                    + row["Prognose_ratio"] * temp_weighted_data["Prognose_ratio"]
                )

                if not np.isnan(weighted_ensemble):
                    data.at[index, "Weighted_ensemble_predicition"] = weighted_ensemble

    data["Average_ensemble_prediction"] = np.nan

    for index, row in data.iterrows():
        current_programme = row["Croho groepeernaam"]
        current_origin = row["Herkomst"]
        current_year = row["Collegejaar"]
        current_week = row["Weeknummer"]
        total = 0
        nr_of_samples = 0

        # Number of samples to calculate average from
        max_nr_of_samples_to_take = 4
        starting_week = 40
        for weeknumber in range(0, max_nr_of_samples_to_take - 1):
            if current_week == starting_week + weeknumber:
                max_nr_of_samples_to_take = weeknumber + 1

        # Calculate average of last three predictions if they exist
        for offset in range(0, max_nr_of_samples_to_take):
            i = index - offset

            offset_year = current_year
            offset_week = current_week - offset
            if offset_week <= 0:
                offset_week += 52
                offset_year -= 1

            # Check if index is in same group of programme and origin
            if (
                i >= 0
                and data.at[i, "Croho groepeernaam"] == current_programme
                and data.at[i, "Herkomst"] == current_origin
                and data.at[i, "Weeknummer"] == offset_week
                and data.at[i, "Collegejaar"] == offset_year
            ):
                total += data.at[i, "Ensemble_prediction"]
                nr_of_samples += 1
        average = total / nr_of_samples
        data.at[index, "Average_ensemble_prediction"] = average
        if row["Weighted_ensemble_predicition"] == -1.0:
            data.at[index, "Weighted_ensemble_predicition"] = average

    return data


### Error measures ###

# MAE ENSEMBLE
# ABS([Aantal studenten] - [Voorspelling Ensemble])

# MAE RATIO
# ABS([Aantal studenten] - [Prognose ratio])

# MAE SARIMA NEW
# ABS([Aantal studenten] - [SARIMA new])

# MAE SARIMA OLD
# ABS([Aantal studenten] - [SARIMA old])

# MAPE ENSEMBLE
# (1/ COUNT([Sheet11])) * sum(ABS(([Aantal studenten] - [N <= 10 ENSEMBLE]) / [Aantal studenten]))

# MAPE RATIO
# (1/ COUNT([Sheet11])) * sum(ABS(([Aantal studenten] - [N <= 10 Ratio]) / [Aantal studenten]))

# MAPE SARIMA NEW
# (1/ COUNT([Sheet11])) * sum(ABS(([Aantal studenten] - [N <= 10 sarima new]) / [Aantal studenten]))

# MAPE SARIMA OLD
# (1/ COUNT([Sheet11])) * sum(ABS(([Aantal studenten] - [N <= 10 sarima old]) / [Aantal studenten]))


def mean_absolute_error(row, key):
    return abs(row["Aantal_studenten"] - row[key])


def mean_absolute_percentage_error(row, key):
    return abs((row["Aantal_studenten"] - row[key]) / row["Aantal_studenten"])


def create_error_columns(data, numerus_fixus_list):
    data["MAE weighted ensemble"] = np.nan
    data["MAE average ensemble"] = np.nan
    data["MAE ensemble"] = np.nan
    data["MAE ratio"] = np.nan
    data["MAE sarima cumulative"] = np.nan
    data["MAE sarima individual"] = np.nan

    data["MAPE weighted ensemble"] = np.nan
    data["MAPE average ensemble"] = np.nan
    data["MAPE ensemble"] = np.nan
    data["MAPE ratio"] = np.nan
    data["MAPE sarima cumulative"] = np.nan
    data["MAPE sarima individual"] = np.nan

    for i, row in data.iterrows():
        if row["Croho groepeernaam"] in numerus_fixus_list:
            continue

        data.at[i, "MAE weighted ensemble"] = mean_absolute_error(
            row, "Weighted_ensemble_predicition"
        )
        data.at[i, "MAE average ensemble"] = mean_absolute_error(
            row, "Average_ensemble_prediction"
        )
        data.at[i, "MAE ensemble"] = mean_absolute_error(row, "Ensemble_prediction")
        data.at[i, "MAE ratio"] = mean_absolute_error(row, "Prognose_ratio")
        data.at[i, "MAE sarima cumulative"] = mean_absolute_error(row, "SARIMA_cumulative")
        data.at[i, "MAE sarima individual"] = mean_absolute_error(row, "SARIMA_individual")

        data.at[i, "MAPE weighted ensemble"] = mean_absolute_percentage_error(
            row, "Weighted_ensemble_predicition"
        )
        data.at[i, "MAPE average ensemble"] = mean_absolute_percentage_error(
            row, "Average_ensemble_prediction"
        )
        data.at[i, "MAPE ensemble"] = mean_absolute_percentage_error(row, "Ensemble_prediction")
        data.at[i, "MAPE ratio"] = mean_absolute_percentage_error(row, "Prognose_ratio")
        data.at[i, "MAPE sarima cumulative"] = mean_absolute_percentage_error(
            row, "SARIMA_cumulative"
        )
        data.at[i, "MAPE sarima individual"] = mean_absolute_percentage_error(
            row, "SARIMA_individual"
        )

    return data


def postprocess(data, weighted_ensemble_data, numerus_fixus_list):
    data = create_ensemble_columns(data, weighted_ensemble_data)
    return create_error_columns(data, numerus_fixus_list)
