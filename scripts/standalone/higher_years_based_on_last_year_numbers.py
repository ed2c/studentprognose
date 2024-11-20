from load_data import load_configuration
import os
import pandas as pd
import numpy as np


class PredictHigherYearsBasedOnLastYearNumbers:
    def __init__(
        self,
        data_student_numbers_first_years,
        data_student_numbers_higher_years,
        data_october,
        data_ratios,
        data_latest,
    ):
        self.data_student_numbers_first_years = data_student_numbers_first_years
        self.data_student_numbers_higher_years = data_student_numbers_higher_years
        self.data_october = data_october
        self.data_ratios = data_ratios
        self.data_latest = data_latest

    def run_predict_with_last_year_numbers(self, predict_year, all_data, skip_years=0, week=0):
        for programme in self.data_october["Groepeernaam Croho"].unique():
            for examtype in self.data_october[
                self.data_october["Groepeernaam Croho"] == programme
            ]["Examentype code"].unique():
                for herkomst in self.data_october[
                    (self.data_october["Groepeernaam Croho"] == programme)
                    & (self.data_october["Examentype code"] == examtype)
                ]["EER-NL-nietEER"].unique():
                    if examtype in ["Bachelor eerstejaars", "Bachelor hogerejaars"]:
                        examtype = "Bachelor"

                    nextyear_higher_years = self.predict_with_last_year_numbers(
                        predict_year, programme, examtype, herkomst, skip_years, week
                    )

                    if not skip_years:
                        all_data.loc[
                            (all_data["Collegejaar"] == predict_year)
                            & (all_data["Croho groepeernaam"] == programme)
                            & (all_data["Herkomst"] == herkomst)
                            & (all_data["Examentype"] == examtype),
                            "Higher_years_prediction_CurrentYear",
                        ] = nextyear_higher_years

                    elif skip_years:
                        all_data.loc[
                            (all_data["Collegejaar"] == predict_year)
                            & (all_data["Croho groepeernaam"] == programme)
                            & (all_data["Herkomst"] == herkomst)
                            & (all_data["Examentype"] == examtype)
                            & (all_data["Weeknummer"] == week),
                            "Higher_years_prediction_CurrentYear",
                        ] = nextyear_higher_years

        return all_data

    def predict_with_last_year_numbers(
        self, predict_year, programme, examtype, herkomst, skip_years, week
    ):
        row = self.data_ratios[
            (self.data_ratios["Collegejaar"] == predict_year)
            & (self.data_ratios["Croho groepeernaam"] == programme)
            & (self.data_ratios["Examentype"] == examtype)
            & (self.data_ratios["Herkomst"] == herkomst)
        ]
        if not row.empty:
            avg_ratio_advancing_to_higher_year = row["Ratio dat doorstroomt"].values[0]
            avg_ratio_dropping_out_in_higher_year = row["Ratio dat uitvalt"].values[0]
        else:
            avg_ratio_advancing_to_higher_year = 0
            avg_ratio_dropping_out_in_higher_year = 0

        if not skip_years:
            currentyear_higher_years = self.data_student_numbers_higher_years[
                (self.data_student_numbers_higher_years["Collegejaar"] == predict_year)
                & (self.data_student_numbers_higher_years["Croho groepeernaam"] == programme)
                & (self.data_student_numbers_higher_years["Examentype"] == examtype)
                & (self.data_student_numbers_higher_years["Herkomst"] == herkomst)
            ]
            if not currentyear_higher_years.empty:
                currentyear_higher_years = currentyear_higher_years["Aantal_studenten"].values[0]
            else:
                currentyear_higher_years = 0

            currentyear_first_years = self.data_student_numbers_first_years[
                (self.data_student_numbers_first_years["Collegejaar"] == predict_year)
                & (self.data_student_numbers_first_years["Croho groepeernaam"] == programme)
                & (self.data_student_numbers_first_years["Examentype"] == examtype)
                & (self.data_student_numbers_first_years["Herkomst"] == herkomst)
            ]
            if not currentyear_first_years.empty:
                currentyear_first_years = currentyear_first_years["Aantal_studenten"].values[0]
            else:
                currentyear_first_years = 0

        if skip_years:
            predicted_row = self.data_latest[
                (self.data_latest["Collegejaar"] == predict_year - skip_years)
                & (self.data_latest["Croho groepeernaam"] == programme)
                & (self.data_latest["Examentype"] == examtype)
                & (self.data_latest["Herkomst"] == herkomst)
                & (self.data_latest["Weeknummer"] == week)
            ]

            currentyear_higher_years = predicted_row["Higher_years_prediction_CurrentYear"]
            if not currentyear_higher_years.empty:
                currentyear_higher_years = currentyear_higher_years.values[0]
            else:
                currentyear_higher_years = 0

            currentyear_first_years = predicted_row["Average_ensemble_prediction"]
            if not currentyear_first_years.empty:
                currentyear_first_years = currentyear_first_years.values[0]
            else:
                currentyear_first_years = 0

        nextyear_higher_years = (
            currentyear_higher_years
            + (currentyear_first_years * avg_ratio_advancing_to_higher_year)
            - (currentyear_higher_years * avg_ratio_dropping_out_in_higher_year)
        )

        return nextyear_higher_years


if __name__ == "__main__":
    configuration = load_configuration("configuration/configuration.json")

    data_student_numbers_first_years = pd.read_excel(
        configuration["paths"]["path_student_count_first-years"]
    )
    data_student_numbers_higher_years = pd.read_excel(
        configuration["paths"]["path_student_count_higher-years"]
    )
    data_october = pd.read_excel(configuration["paths"]["path_october"])
    data_ratios = pd.read_excel(configuration["paths"]["path_ratios"])
    data_latest = pd.read_excel(configuration["paths"]["path_latest"])

    predict_higher_years_based_on_last_year_numbers = PredictHigherYearsBasedOnLastYearNumbers(
        data_student_numbers_first_years,
        data_student_numbers_higher_years,
        data_october,
        data_ratios,
        data_latest,
    )

    predict_years = [2024, 2025]
    skip_years = 0
    week = 0

    for predict_year in predict_years:
        print(f"Predicting year {predict_year}")
        data_latest = (
            predict_higher_years_based_on_last_year_numbers.run_predict_with_last_year_numbers(
                predict_year, data_latest, skip_years, week
            )
        )

    print("Adding MAE and MAPE errors (if applicable)")
    data_latest["MAE_higher_years_CurrentYear"] = data_latest.apply(
        lambda row: (
            abs(row["Aantal_studenten_higher_years"] - row["Higher_years_prediction_CurrentYear"])
            if pd.notna(row["Higher_years_prediction_CurrentYear"])
            and pd.notna(row["Aantal_studenten_higher_years"])
            else np.nan
        ),
        axis=1,
    )
    data_latest["MAPE_higher_years_CurrentYear"] = data_latest.apply(
        lambda row: (
            abs(
                (row["Aantal_studenten_higher_years"] - row["Higher_years_prediction_CurrentYear"])
                / row["Aantal_studenten_higher_years"]
            )
            if pd.notna(row["Higher_years_prediction_CurrentYear"])
            and pd.notna(row["Aantal_studenten_higher_years"])
            and row["Aantal_studenten_higher_years"] != 0
            else np.nan
        ),
        axis=1,
    )

    data_latest.sort_values(
        by=["Croho groepeernaam", "Examentype", "Collegejaar", "Weeknummer", "Herkomst"],
        inplace=True,
        ignore_index=True,
    )

    print("Saving output...")
    CWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outfile = os.path.join(CWD, "data/output/output_higher-years.xlsx")

    data_latest.to_excel(outfile)
