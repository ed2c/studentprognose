import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor


class HigherYears:
    def __init__(
        self,
        data_student_numbers_first_years,
        data_student_numbers_higher_years,
        data_student_numbers_volume,
        configuration,
    ):
        self.data_student_numbers_first_years = data_student_numbers_first_years
        self.data_student_numbers_higher_years = data_student_numbers_higher_years
        self.data_student_numbers_volume = data_student_numbers_volume
        self.numerus_fixus_list = configuration["numerus_fixus"]

    def predict_nr_of_students(
        self, first_year_data, all_data, predict_year, predict_week, skip_years
    ):
        """
        Predicts number of higher year and volume students by a ratio model and an XGBoost model.

        Args:
            first_year_data (pd.DataFrame): Data about first year students.
            all_data (pd.DataFrame): Extended first_year_data. This dataframe will be filled in.
            predict_year (int): The year to be predicted
            predict_week (int): The week to be predicted
            skip_years (int): The years to be skipped if we want to predict more time ahead.

        Returns:
            pd.DataFrame: A DataFrame including the higher_year and volume predictions and MAE en
            MAPE error.
        """

        # data_student_numbers_first_years = self.data_student_numbers_first_years.copy(deep=True)
        # data_student_numbers_higher_years = self.data_student_numbers_higher_years.copy(deep=True)
        # data_student_numbers_volume = self.data_student_numbers_volume.copy(deep=True)

        self.predict_year = predict_year
        self.predict_week = predict_week
        self.skip_years = skip_years

        combined_data_student_numbers = self.preprocess(first_year_data)
        self.predict_with_ratio(combined_data_student_numbers)
        all_data = self.predict_with_xgboost(combined_data_student_numbers, all_data)
        all_data = self.fill_in_final_dataframe(all_data)

        return all_data

    def preprocess(self, first_year_data):
        prediction_data = first_year_data[
            (first_year_data["Collegejaar"] == self.predict_year)
            & (first_year_data["Weeknummer"] == self.predict_week)
        ]

        data_student_numbers_first_years = self.data_student_numbers_first_years

        # If there is a row known in the prediction data with the same 'Croho groepeernaam' and 'Herkomst' as
        # the sample row, fill in the 'Weighted_ensemble_prediction' if the programme, herkomst and
        # collegejaar is already known in data_student_numbers_first_years. Otherwise add a new row.
        for _, row in data_student_numbers_first_years.iterrows():
            pred = prediction_data[
                (prediction_data["Croho groepeernaam"] == row["Croho groepeernaam"])
                & (prediction_data["Herkomst"] == row["Herkomst"])
            ]
            if len(pred) > 0:
                if (
                    len(
                        data_student_numbers_first_years.loc[
                            (
                                data_student_numbers_first_years["Croho groepeernaam"]
                                == row["Croho groepeernaam"]
                            )
                            & (data_student_numbers_first_years["Herkomst"] == row["Herkomst"])
                            & (
                                data_student_numbers_first_years["Collegejaar"]
                                == self.predict_year
                            )
                        ]
                    )
                    == 0
                ):
                    data_student_numbers_first_years = pd.concat(
                        [
                            data_student_numbers_first_years,
                            pred[
                                [
                                    "Collegejaar",
                                    "Croho groepeernaam",
                                    "Herkomst",
                                    "Weighted_ensemble_prediction",
                                ]
                            ].rename(columns={"Weighted_ensemble_prediction": "Aantal_studenten"}),
                        ],
                        ignore_index=True,
                    )
                else:
                    data_student_numbers_first_years.loc[
                        (
                            data_student_numbers_first_years["Croho groepeernaam"]
                            == row["Croho groepeernaam"]
                        )
                        & (data_student_numbers_first_years["Herkomst"] == row["Herkomst"])
                        & (data_student_numbers_first_years["Collegejaar"] == self.predict_year),
                        "Aantal_studenten",
                    ] = pred["Weighted_ensemble_prediction"].iloc[0]
            else:
                data_student_numbers_first_years.loc[
                    (
                        data_student_numbers_first_years["Croho groepeernaam"]
                        == row["Croho groepeernaam"]
                    )
                    & (data_student_numbers_first_years["Herkomst"] == row["Herkomst"])
                    & (data_student_numbers_first_years["Collegejaar"] == self.predict_year),
                    "Aantal_studenten",
                ] = 0

        # Merge the first year data with the higher years.
        combined_data_student_numbers = data_student_numbers_first_years.merge(
            self.data_student_numbers_higher_years,
            on=["Croho groepeernaam", "Herkomst", "Collegejaar"],
            how="left",
        )

        # We have two columns with studentcounts now (one is first year student count and the other one
        # is about higher years). We rename it to be more recognisable.
        combined_data_student_numbers = combined_data_student_numbers.rename(
            columns={
                "Aantal_studenten_x": "Aantal_studenten_f",
                "Aantal_studenten_y": "Aantal_studenten_h",
            }
        )

        self.data_student_numbers_first_years = data_student_numbers_first_years
        return combined_data_student_numbers

    def predict_with_ratio(self, combined_data_student_numbers):
        # Ratio prediction
        if self.predict_year <= 2021:
            ratio_starting_year = np.sort(combined_data_student_numbers["Collegejaar"].unique())[0]
        else:
            ratio_starting_year = 2021
        ratio_ending_year = self.predict_year - self.skip_years - 1

        ratio_data = combined_data_student_numbers[
            (combined_data_student_numbers["Collegejaar"] >= ratio_starting_year)
            & (combined_data_student_numbers["Collegejaar"] <= ratio_ending_year)
        ]
        ratio_data["Ratio"] = ratio_data["Aantal_studenten_h"].divide(
            ratio_data["Aantal_studenten_f"]
        )
        ratio_data["Ratio"] = ratio_data["Ratio"].replace(np.inf, np.nan)
        # ratio_data[ratio_data["Croho groepeernaam"] == "M Geneeskunde"]

        ratio = (
            ratio_data[["Croho groepeernaam", "Herkomst", "Ratio", "Examentype"]]
            .groupby(["Croho groepeernaam", "Herkomst", "Examentype"], as_index=False)
            .mean()
        )

        self.ratio = ratio

    def predict_with_xgboost(self, combined_data_student_numbers, all_data):
        # XGBoost prediction
        XGB_data = combined_data_student_numbers

        # all_examtypes = XGB_data["Examentype"].unique()
        all_examtypes = [
            "Bachelor",
            "Master",
        ]  # Pre-master is not yet added to student_count_higher-years.xlsx

        for nf_programme in self.numerus_fixus_list:
            train = XGB_data[
                (XGB_data["Examentype"] == "Bachelor")
                & (XGB_data["Croho groepeernaam"] == nf_programme)
                & (XGB_data["Collegejaar"] < self.predict_year - self.skip_years)
            ]
            test = XGB_data[
                (XGB_data["Examentype"] == "Bachelor")
                & (XGB_data["Croho groepeernaam"] == nf_programme)
                & (XGB_data["Collegejaar"] == self.predict_year)
            ]

            test_sorted, predictie = self.xgboost_fit_predict(train, test)
            all_data = self.combination(test_sorted, predictie, all_data)

        for examtype in all_examtypes:
            train = XGB_data[
                (XGB_data["Examentype"] == examtype)
                & (~XGB_data["Croho groepeernaam"].isin(self.numerus_fixus_list))
                & (XGB_data["Collegejaar"] < self.predict_year - self.skip_years)
            ]
            test = XGB_data[
                (XGB_data["Examentype"] == examtype)
                & (~XGB_data["Croho groepeernaam"].isin(self.numerus_fixus_list))
                & (XGB_data["Collegejaar"] == self.predict_year)
            ]

            test_sorted, predictie = self.xgboost_fit_predict(train, test)
            all_data = self.combination(test_sorted, predictie, all_data)

        return all_data

    def xgboost_fit_predict(self, train, test):
        if len(test) == 0:
            return test, []

        train = train.fillna(0)
        test = test.fillna(0)

        train = train.drop_duplicates()

        X_train = train.drop(["Aantal_studenten_h"], axis=1)
        y_train = train.pop("Aantal_studenten_h")

        test = test.sort_values(by=["Croho groepeernaam", "Herkomst"])

        numeric_cols = ["Collegejaar", "Aantal_studenten_f"]
        categorical_cols = ["Croho groepeernaam", "Herkomst", "Examentype"]

        numeric_transformer = "passthrough"  # No transformation for numeric columns
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        # Create the column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_cols),
                ("categorical", categorical_transformer, categorical_cols),
            ]
        )

        X_train = preprocessor.fit_transform(X_train)
        preprocessed_test = preprocessor.transform(test)

        # model = XGBRegressor(learning_rate=0.5301857237491415)
        model = XGBRegressor(learning_rate=0.1)

        model.fit(X_train, y_train)

        predictie = model.predict(preprocessed_test)

        return test, predictie

    def combination(self, test, predictie, all_data):
        # Combination
        i = 0
        for _, row in test.iterrows():
            programme = row["Croho groepeernaam"]
            origin = row["Herkomst"]
            first_year_row = self.data_student_numbers_first_years[
                (self.data_student_numbers_first_years["Croho groepeernaam"] == programme)
                & (self.data_student_numbers_first_years["Herkomst"] == origin)
                & (self.data_student_numbers_first_years["Collegejaar"] == self.predict_year)
            ]
            current_ratio = self.ratio[
                (self.ratio["Croho groepeernaam"] == programme)
                & (self.ratio["Herkomst"] == origin)
            ]

            final_prediction = predictie[i]
            i += 1

            all_data.loc[
                (all_data["Collegejaar"] == self.predict_year)
                & (all_data["Weeknummer"] == self.predict_week)
                & (all_data["Croho groepeernaam"] == programme)
                & (all_data["Herkomst"] == origin),
                "Higher_years_prediction_XGBoost",
            ] = final_prediction

            if len(first_year_row) > 0 and len(current_ratio) > 0:
                ratio_prediction = (
                    first_year_row["Aantal_studenten"].iloc[0] * current_ratio["Ratio"].iloc[0]
                )

                all_data.loc[
                    (all_data["Collegejaar"] == self.predict_year)
                    & (all_data["Weeknummer"] == self.predict_week)
                    & (all_data["Croho groepeernaam"] == programme)
                    & (all_data["Herkomst"] == origin),
                    "Higher_years_prediction_Ratio",
                ] = ratio_prediction

                # Numbers obtained by optimizing with optuna
                final_prediction = final_prediction * 0.84 + ratio_prediction * 0.16

            all_data.loc[
                (all_data["Collegejaar"] == self.predict_year)
                & (all_data["Weeknummer"] == self.predict_week)
                & (all_data["Croho groepeernaam"] == programme)
                & (all_data["Herkomst"] == origin),
                "Higher_years_prediction",
            ] = final_prediction

            # Calculate volume based on both years prediction and add it to the data
            prediction_label = "Weighted_ensemble_prediction"
            if self.skip_years > 0:
                prediction_label = "Skip_prediction"

            volume_prediction = (
                final_prediction
                + all_data[
                    (all_data["Collegejaar"] == self.predict_year)
                    & (all_data["Weeknummer"] == self.predict_week)
                    & (all_data["Croho groepeernaam"] == programme)
                    & (all_data["Herkomst"] == origin)
                ][prediction_label]
            )

            all_data.loc[
                (all_data["Collegejaar"] == self.predict_year)
                & (all_data["Weeknummer"] == self.predict_week)
                & (all_data["Croho groepeernaam"] == programme)
                & (all_data["Herkomst"] == origin),
                "Volume_prediction",
            ] = volume_prediction

        return all_data

    def fill_in_final_dataframe(self, all_data):

        if "Aantal_studenten_higher_years" in all_data.columns:
            all_data = all_data.drop(["Aantal_studenten_higher_years"], axis=1)
        if "Aantal_studenten_volume" in all_data.columns:
            all_data = all_data.drop(["Aantal_studenten_volume"], axis=1)

        all_data = all_data.merge(
            self.data_student_numbers_higher_years.rename(
                columns={"Aantal_studenten": "Aantal_studenten_higher_years"}
            ),
            on=["Croho groepeernaam", "Herkomst", "Collegejaar"],
            how="left",
        )
        all_data = all_data.merge(
            self.data_student_numbers_volume.rename(
                columns={"Aantal_studenten": "Aantal_studenten_volume"}
            ),
            on=["Croho groepeernaam", "Herkomst", "Collegejaar"],
            how="left",
        )

        all_data["MAE_higher_years_XGBoost"] = abs(
            all_data["Higher_years_prediction_XGBoost"] - all_data["Aantal_studenten_higher_years"]
        )
        if "Higher_years_prediction_Ratio" in all_data.columns:
            all_data["MAE_higher_years_Ratio"] = abs(
                all_data["Higher_years_prediction_Ratio"]
                - all_data["Aantal_studenten_higher_years"]
            )
        all_data["MAE_higher_years"] = abs(
            all_data["Higher_years_prediction"] - all_data["Aantal_studenten_higher_years"]
        )
        all_data["MAE_volume"] = abs(
            all_data["Volume_prediction"] - all_data["Aantal_studenten_volume"]
        )

        all_data["MAPE_higher_years_XGBoost"] = abs(
            (
                all_data["Higher_years_prediction_XGBoost"]
                - all_data["Aantal_studenten_higher_years"]
            )
            / all_data["Aantal_studenten_higher_years"]
        )
        if "Higher_years_prediction_Ratio" in all_data.columns:
            all_data["MAPE_higher_years_Ratio"] = abs(
                (
                    all_data["Higher_years_prediction_Ratio"]
                    - all_data["Aantal_studenten_higher_years"]
                )
                / all_data["Aantal_studenten_higher_years"]
            )
        all_data["MAPE_higher_years"] = abs(
            (all_data["Higher_years_prediction"] - all_data["Aantal_studenten_higher_years"])
            / all_data["Aantal_studenten_higher_years"]
        )
        all_data["MAPE_volume"] = abs(
            (all_data["Volume_prediction"] - all_data["Aantal_studenten_volume"])
            / all_data["Aantal_studenten_volume"]
        )

        columns_to_consider = list(all_data.columns)
        for column in ["level_0", "index"]:
            columns_to_consider.remove(column)
        all_data = all_data.drop_duplicates(subset=columns_to_consider)

        return all_data
