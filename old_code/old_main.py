import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from configparser import ConfigParser

import os
import sys
import datetime

from scripts.preprocess import read_and_preprocess, vooraanmeldingen_joinen
from scripts.helper import get_max_week, get_weeks_list, get_max_week_from_weeks
from scripts.predict_sarima import make_full_week_prediction
from scripts.transform_data import create_total_file, replace_latest_data
from scripts.load_data import load_data, load_configuration, load_data_old
from scripts.postprocess import postprocess

CWD = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = "output"

tqdm.pandas()
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Predictor:
    def __init__(self, years, weeks) -> None:
        # Variables
        self.years = years
        self.weeks = weeks
        self.avg_ratio_between = (2021, 2023)

        self.configuration = load_configuration()
        self.numerus_fixus_list = list(self.configuration["numerus_fixus"].keys())

        # Load data into dataframes
        (
            self.data_individual,
            self.data_cumulative,
            self.data_student_numbers,
            self.data_latest,
            self.data_distances,
            self.data_weighted_ensemble,
        ) = load_data_old()

        # Preprocess dataframes
        self.data_individual = read_and_preprocess(
            self.data_individual, self.data_distances, self.numerus_fixus_list
        )
        self.data_cumulative = vooraanmeldingen_joinen(self.data_cumulative)

        self.data_individual = self.data_individual.merge(
            self.data_cumulative,
            on=[
                "Croho groepeernaam",
                "Collegejaar",
                "Faculteit",
                "Examentype",
                "Weeknummer",
                "Herkomst",
            ],
            how="left",
        )

    def predict(self, save_backup=True, replace_input=True):
        if save_backup:
            output_directory_path = os.path.join(CWD, "data", "output")
            output_path = os.path.join(CWD, "data", "output", "total_backup.xlsx")
            if not os.path.exists(output_directory_path):
                os.makedirs(output_directory_path)
            self.data_latest.to_excel(output_path, index=False)

        MAX_YEAR = self.data_individual["Collegejaar"].max()

        prediction_df = pd.DataFrame(
            columns=[
                "Collegejaar",
                "Faculteit",
                "Examentype",
                "Herkomst",
                "Croho groepeernaam",
                "Weeknummer",
                "SARIMA",
            ]
        )

        for year in self.years:
            if year == MAX_YEAR:
                max_week = get_max_week_from_weeks(
                    self.data_individual[self.data_individual.Collegejaar == year]["Weeknummer"]
                )
            else:
                if year == 2021:
                    max_week = 38
                else:
                    max_week = get_max_week_from_weeks(
                        self.data_individual[self.data_individual.Collegejaar == year][
                            "Weeknummer"
                        ]
                    )

            if self.weeks == "All":
                self.weeks = get_weeks_list(38)

            for week in self.weeks:

                print("Now running week " + str(week))

                if week in [40, 41] and year == 2021:
                    pass

                else:

                    only_individual = False
                    if week not in self.data_cumulative["Weeknummer"]:
                        only_individual = True

                    predictions = make_full_week_prediction(
                        self.data_individual,
                        year,
                        MAX_YEAR,
                        week,
                        max_week,
                        self.data_cumulative,
                        self.data_student_numbers,
                        only_individual,
                        self.numerus_fixus_list,
                    )

                    NFs = self.configuration["numerus_fixus"]

                    for year in predictions["Collegejaar"].unique():
                        for week in predictions["Weeknummer"].unique():
                            for nf in NFs:
                                nf_data = predictions[
                                    (predictions["Collegejaar"] == year)
                                    & (predictions["Weeknummer"] == week)
                                    & (predictions["Croho groepeernaam"] == nf)
                                ]
                                if np.sum(nf_data["SARIMA_individual"]) > NFs[nf]:
                                    predictions.loc[
                                        (predictions["Collegejaar"] == year)
                                        & (predictions["Weeknummer"] == week)
                                        & (predictions["Croho groepeernaam"] == nf)
                                        & (predictions["Herkomst"] == "NL"),
                                        "SARIMA_individual",
                                    ] = nf_data[nf_data["Herkomst"] == "NL"][
                                        "SARIMA_individual"
                                    ] - (
                                        np.sum(nf_data["SARIMA_individual"]) - NFs[nf]
                                    )

                                if np.sum(nf_data["SARIMA_cumulative"]) > NFs[nf]:
                                    predictions.loc[
                                        (predictions["Collegejaar"] == year)
                                        & (predictions["Weeknummer"] == week)
                                        & (predictions["Croho groepeernaam"] == nf)
                                        & (predictions["Herkomst"] == "NL"),
                                        "SARIMA_cumulative",
                                    ] = nf_data[nf_data["Herkomst"] == "NL"][
                                        "SARIMA_cumulative"
                                    ] - (
                                        np.sum(nf_data["SARIMA_cumulative"]) - NFs[nf]
                                    )

                    prediction_df = replace_latest_data(prediction_df, predictions)

                    try:
                        prediction_df.to_excel(
                            SAVE_PATH + "\\Voorspellingen-{}.xlsx".format(str(week))
                        )
                    except:
                        pass

            prediction_df = prediction_df.merge(
                self.data_cumulative,
                on=["Croho groepeernaam", "Collegejaar", "Weeknummer", "Herkomst"],
                how="left",
            )

            data_total = create_total_file(
                prediction_df, self.data_cumulative, self.data_student_numbers
            )

            data_total["Faculteit"].replace(
                {
                    "LET": "FdL",
                    "SOW": "FSW",
                    "RU": "FdM",
                    "MAN": "FdM",
                    "NWI": "FNWI",
                    "MED": "FMW",
                    "FTR": "FFTR",
                    "JUR": "FdR",
                },
                inplace=True,
            )

            output_path = os.path.join(CWD, "data", "output", "output_prelim.xlsx")
            data_total.to_excel(output_path, index=False)

            data_total = self._predict_with_ratio(data_total)

            self.data_latest = replace_latest_data(self.data_latest, data_total)

            output_path = os.path.join(CWD, "data", "output", "output_prefinal.xlsx")
            self.data_latest.to_excel(output_path, index=False)

            if replace_input:
                output_path = os.path.join(CWD, "data", "input", "totaal.xlsx")
                self.data_latest.to_excel(output_path, index=False)

            # Perform postprocessing by adding ensemble and error columns
            self.data_latest = postprocess(
                self.data_latest, self.data_weighted_ensemble, self.numerus_fixus_list
            )

            output_path = os.path.join(CWD, "data", "output", "output_final.xlsx")
            self.data_latest.to_excel(output_path, index=False)

    def _predict_with_ratio(self, data_vooraanmeldingen_nieuw):
        data_vooraanmeldingen = self.data_cumulative[
            [
                "Collegejaar",
                "Weeknummer",
                "Croho groepeernaam",
                "Herkomst",
                "Ongewogen vooraanmelders",
                "Inschrijvingen",
            ]
        ]
        data_vooraanmeldingen["Ongewogen vooraanmelders"] = data_vooraanmeldingen[
            "Ongewogen vooraanmelders"
        ].astype("float")
        data_vooraanmeldingen["Aanmeldingen"] = (
            data_vooraanmeldingen["Ongewogen vooraanmelders"]
            + data_vooraanmeldingen["Inschrijvingen"]
        )
        data_vooraanmeldingen = data_vooraanmeldingen[
            (data_vooraanmeldingen["Collegejaar"] >= self.avg_ratio_between[0])
            & (data_vooraanmeldingen["Collegejaar"] <= self.avg_ratio_between[1])
        ]

        data_merged = pd.merge(
            left=data_vooraanmeldingen,
            right=self.data_student_numbers,
            on=["Collegejaar", "Croho groepeernaam", "Herkomst"],
            how="left",
        )
        data_merged["Avg_ratio"] = data_merged["Aanmeldingen"].divide(
            data_merged["Aantal_studenten"]
        )
        data_merged["Avg_ratio"] = data_merged["Avg_ratio"].replace(np.inf, np.nan)

        avg_ratios = (
            data_merged[["Croho groepeernaam", "Herkomst", "Weeknummer", "Avg_ratio"]]
            .groupby(["Croho groepeernaam", "Herkomst", "Weeknummer"], as_index=False)
            .mean()
        )

        data_vooraanmeldingen_nieuw = data_vooraanmeldingen_nieuw.rename(
            columns={"Aantal eerstejaarsopleiding": "EOI_vorigjaar"}
        )
        data_vooraanmeldingen_nieuw["Ongewogen vooraanmelders"] = data_vooraanmeldingen_nieuw[
            "Ongewogen vooraanmelders"
        ].astype("float")
        data_vooraanmeldingen_nieuw["Aanmelding"] = (
            data_vooraanmeldingen_nieuw["Ongewogen vooraanmelders"]
            + data_vooraanmeldingen_nieuw["Inschrijvingen"]
        )

        data_vooraanmeldingen_nieuw["Ratio"] = (data_vooraanmeldingen_nieuw["Aanmelding"]).divide(
            data_vooraanmeldingen_nieuw["Aantal_studenten"]
        )
        data_vooraanmeldingen_nieuw["Ratio"] = data_vooraanmeldingen_nieuw["Ratio"].replace(
            np.inf, np.nan
        )

        data_vooraanmeldingen_nieuw = pd.merge(
            left=data_vooraanmeldingen_nieuw,
            right=avg_ratios,
            on=["Croho groepeernaam", "Herkomst", "Weeknummer"],
        )

        for i, row in data_vooraanmeldingen_nieuw.iterrows():
            if row["Avg_ratio"] != 0:
                data_vooraanmeldingen_nieuw.at[i, "Prognose_ratio"] = (
                    row["Aanmelding"] / row["Avg_ratio"]
                )
        # data_vooraanmeldingen_nieuw['Prognose_ratio'] = data_vooraanmeldingen_nieuw['Aanmelding'].divide(data_vooraanmeldingen_nieuw['Avg_ratio'])

        NFs = self.configuration["numerus_fixus"]
        for year in data_vooraanmeldingen_nieuw["Collegejaar"].unique():
            for week in data_vooraanmeldingen_nieuw["Weeknummer"].unique():
                for nf in NFs:
                    nf_data = data_vooraanmeldingen_nieuw[
                        (data_vooraanmeldingen_nieuw["Collegejaar"] == year)
                        & (data_vooraanmeldingen_nieuw["Weeknummer"] == week)
                        & (data_vooraanmeldingen_nieuw["Croho groepeernaam"] == nf)
                    ]

                    if np.sum(nf_data["Prognose_ratio"]) > NFs[nf]:
                        data_vooraanmeldingen_nieuw.loc[
                            (data_vooraanmeldingen_nieuw["Collegejaar"] == year)
                            & (data_vooraanmeldingen_nieuw["Weeknummer"] == week)
                            & (data_vooraanmeldingen_nieuw["Croho groepeernaam"] == nf)
                            & (data_vooraanmeldingen_nieuw["Herkomst"] == "NL"),
                            "Prognose_ratio",
                        ] = nf_data[nf_data["Herkomst"] == "NL"]["Prognose_ratio"] - (
                            np.sum(nf_data["Prognose_ratio"]) - NFs[nf]
                        )
                        # nf_data[nf_data["Herkomst"] == "NL"]["Prognose_ratio"] = nf_data[nf_data["Herkomst"] == "NL"]["Prognose_ratio"] - (
                        #     np.sum(nf_data['Prognose_ratio']) - NFs[nf]
                        # )

        # for i, row in data_vooraanmeldingen_nieuw.iterrows():
        #     if row["Croho groepeernaam"] in self.numerus_fixus_list:
        #         if row["Prognose_ratio"] > self.configuration["numerus_fixus"][row["Croho groepeernaam"]]:
        #             data_vooraanmeldingen_nieuw.at[i, "Prognose_ratio"] = self.configuration["numerus_fixus"][row["Croho groepeernaam"]]

        data_vooraanmeldingen_nieuw = data_vooraanmeldingen_nieuw.rename(
            columns={"Avg_ratio": "Average_Ratio"}
        )

        return data_vooraanmeldingen_nieuw


if __name__ == "__main__":
    get_weeks = False
    get_years = False

    weeks = []
    years = []

    # First arguments is always the name of the python script
    for i in range(1, len(sys.argv)):
        arg = sys.argv[i]

        if get_weeks and arg.isnumeric():
            weeks.append(int(arg))
        elif get_years and arg.isnumeric():
            years.append(int(arg))

        if arg == "-w" or arg == "-W" or arg == "-week":
            get_weeks = True
            get_years = False
        elif arg == "-y" or arg == "-Y" or arg == "-year":
            get_weeks = False
            get_years = True

    weeks_specified = True
    if weeks == []:
        weeks_specified = False
        current_week = datetime.date.today().isocalendar()[1]
        # Max of 52 weeks, week 53 is an edge case where the user should manually input data
        if current_week > 52:
            print("Current week is week 53, check what weeknumber should be used")
            print("Now predicting for week 52")
            current_week = 52
        weeks = [current_week]

    if years == []:
        current_year = datetime.date.today().year

        if not weeks_specified and weeks[0] >= 40:
            current_year += 1

        years = [current_year]

    print("Predicting for years: ", years, " and weeks: ", weeks)

    pred = Predictor(years, weeks)
    pred.predict()
