import os as os
import pandas as pd
import numpy as np
from load_data import load_configuration


def recent_year_count(data, student_count_first_years, student_count_higher_years):

    def _mean_absolute_error(actual, predicted):
        return abs(actual - predicted)

    def _mean_absolute_percentage_error(actual, predicted):
        return abs((actual - predicted) / actual) if actual != 0 else np.nan

    def _calculate_errors(row):
        actual = row["Aantal_studenten"]
        errors = {}
        for key in [
            "Weighted_ensemble_prediction",
            "Average_ensemble_prediction",
            "Ensemble_prediction",
            "Prognose_ratio",
            "SARIMA_cumulative",
            "SARIMA_individual",
        ]:
            predicted = convert_nan_to_zero(row[key])
            errors[f"MAE_{key}"] = _mean_absolute_error(actual, predicted)
            errors[f"MAPE_{key}"] = _mean_absolute_percentage_error(actual, predicted)
        return errors

    def convert_nan_to_zero(number):
        if pd.isnull(number):
            return 0
        else:
            return number

    output2024 = output[output.Collegejaar == 2024].drop(
        columns=["Aantal_studenten", "Aantal_studenten_higher_years"]
    )
    herkomsten = output2024["Herkomst"].unique()
    weeknummers = output2024["Weeknummer"].unique()
    for programme in output2024["Croho groepeernaam"].unique():
        for examtype in output2024[output2024["Croho groepeernaam"] == programme][
            "Examentype"
        ].unique():
            faculteit = output2024[
                (output2024["Croho groepeernaam"] == programme)
                & (output2024["Examentype"] == examtype)
            ]["Faculteit"].unique()[0]
            for week in weeknummers:
                for herkomst in herkomsten:
                    if output2024[
                        (output2024["Croho groepeernaam"] == programme)
                        & (output2024["Examentype"] == examtype)
                        & (output2024["Weeknummer"] == week)
                        & (output2024["Herkomst"] == herkomst)
                    ].empty:
                        new_row = {
                            "Croho groepeernaam": programme,
                            "Examentype": examtype,
                            "Faculteit": faculteit,
                            "Weeknummer": week,
                            "Herkomst": herkomst,
                            "Collegejaar": 2024,
                            # Columns with 0 values
                            "SARIMA_cumulative": 0,
                            "SARIMA_individual": 0,
                            "Gewogen vooraanmelders": 0,
                            "Ongewogen vooraanmelders": 0,
                            "Aantal aanmelders met 1 aanmelding": 0,
                            "Inschrijvingen": 0,
                            "Prognose_ratio": 0,
                            "Ratio": 0,
                            "Aanmelding": 0,
                            "Average_Ratio": 0,
                            "Ensemble_prediction": 0,
                            "Weighted_ensemble_prediction": 0,
                            "Average_ensemble_prediction": 0,
                        }

                        # Append the new row to the DataFrame
                        output2024 = output2024._append(new_row, ignore_index=True)
    # Set Prognose_ratio to NaN for Weeknummer 39 in 2024
    output2024.sort_values(
        by=["Croho groepeernaam", "Examentype", "Collegejaar", "Weeknummer", "Herkomst"],
        inplace=True,
    )
    output2024.loc[output2024["Weeknummer"] == 39, "Prognose_ratio"] = np.nan
    firstyear_studenten2024 = student_count_first_years[
        student_count_first_years.Collegejaar == 2024
    ]
    data = pd.merge(
        output2024,
        firstyear_studenten2024,
        how="left",
        on=["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"],
    )
    higheryear_studenten2024 = student_count_higher_years[
        student_count_higher_years.Collegejaar == 2024
    ]
    higheryear_studenten2024 = higheryear_studenten2024.rename(
        columns={"Aantal_studenten": "Aantal_studenten_higher_years"}
    )

    data = pd.merge(
        data,
        higheryear_studenten2024,
        how="left",
        on=["Collegejaar", "Croho groepeernaam", "Herkomst", "Examentype"],
    )

    errors_df = data.apply(_calculate_errors, axis=1, result_type="expand")
    data = data.drop(columns=errors_df.columns)
    data = pd.concat([data, errors_df], axis=1)
    data_wo2024 = output[output.Collegejaar != 2024]
    data = pd.concat([data_wo2024, data])

    return data


if __name__ == "__main__":
    configuration = load_configuration("configuration/configuration.json")
    output = pd.read_excel(configuration["paths"]["path_latest"])
    student_count_first_years = pd.read_excel(
        configuration["paths"]["path_student_count_first-years"]
    )
    student_count_higher_years = pd.read_excel(
        configuration["paths"]["path_student_count_higher-years"]
    )

    result = recent_year_count(output, student_count_first_years, student_count_higher_years)

    CWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outfile = os.path.join(CWD, "data/output/totaal.xlsx")
    result.to_excel(outfile)
