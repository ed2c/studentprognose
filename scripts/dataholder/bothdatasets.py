from scripts.dataholder.superclass import *
from scripts.dataholder.individual import *
from scripts.dataholder.cumulative import *
from scripts.dataholder.helpermethods import *


class BothDatasets(Superclass):
    def __init__(
        self,
        data_individual,
        data_cumulative,
        data_distances,
        data_studentcount,
        configuration,
        helpermethods_initialise_material,
        years,
    ):
        super().__init__(configuration, helpermethods_initialise_material)

        # Initialize the individual and cumulative datasets
        self.individual = Individual(
            data_individual, data_distances, configuration, helpermethods_initialise_material
        )
        self.cumulative = Cumulative(
            data_cumulative, data_studentcount, configuration, helpermethods_initialise_material
        )

        # Check if the selected years are present in the individual dataset
        if not all(
            year in self.individual.data_individual["Collegejaar"].unique() for year in years
        ):
            raise ValueError(
                f"Selected years {years} not found in individual dataset. Proceeding with cumulative dataset."
            )
        # Set the years
        self.years = years

    def preprocess(self):
        print("Preprocessing individual data...")
        self.individual.preprocess()
        print("Preprocessing cumulative data...")
        return self.cumulative.preprocess()

    def predict_nr_of_students(self, predict_year, predict_week, skip_years=0):
        """
        Predicts the number of students by first predicting the SARIMA_individual and then the
        SARIMA_cumulative.

        Args:
            predict_year (int): The year to be predicted
            predict_week (int): The week to be predicted
            skip_year (int): The years to be skipped if we want to predict more time ahead.

        Returns:
            pd.DataFrame: A DataFrame including the SARIMA_individual, SARIMA_cumulative and predicted
            pre-applicants.
        """

        self.individual.data_individual = self.individual.data_individual_backup.copy(deep=True)
        self.cumulative.data_cumulative = self.cumulative.data_cumulative_backup.copy(deep=True)

        self.set_year_week(predict_year, predict_week, self.cumulative.data_cumulative)
        self.individual.set_year_week(predict_year, predict_week, self.individual.data_individual)
        self.cumulative.set_year_week(predict_year, predict_week, self.cumulative.data_cumulative)

        self.individual.data_individual = self.individual.data_individual.merge(
            self.cumulative.data_cumulative,
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

        # Call predict_applicant first
        print("Predicting preapplicants...")
        predicties = self.individual.predict_applicant(
            self.individual.data_individual, self.cumulative.data_cumulative
        )
        self.individual.data_individual.loc[
            (self.individual.data_individual["Collegejaar"] == self.predict_year)
            & (
                self.individual.data_individual["Weeknummer"].isin(
                    get_weeks_list(self.predict_week)
                )
            ),
            "Inschrijvingen_predictie",
        ] = predicties

        # Transform individual data to be used for sarima
        self.individual.transform_data_individual()

        temp_data_individual = self.individual.data_individual.copy(deep=True)
        temp_data_individual["Weeknummer"] = self.individual.data_individual["Weeknummer"].astype(
            int
        )

        # Create exogenous data for sarima with the individual dataset
        self.data_exog = temp_data_individual.merge(
            self.cumulative.data_cumulative,
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

        self.individual.data_individual = transform_data(
            self.individual.data_individual, "Cumulative_sum_within_year"
        )

        # Prepare the data_cumulative and use it to initialise full_data (data from which the test and
        # training data will be filtered for xgboost) and data_to_predict
        self.cumulative.prepare_data()

        full_data = self.cumulative.get_transformed_data(
            self.cumulative.data_cumulative.copy(deep=True)
        )
        full_data["39"] = 0

        self.skip_years = skip_years

        # Filter all the cumulative data on predict year, predict week, programme and herkomst to
        # obtain the data_to_predict.
        data_to_predict = self.cumulative.data_cumulative[
            (self.cumulative.data_cumulative["Collegejaar"] == self.predict_year)
            & (self.cumulative.data_cumulative["Weeknummer"] == self.predict_week)
        ]
        if self.programme_filtering != []:
            data_to_predict = data_to_predict[
                (data_to_predict["Croho groepeernaam"].isin(self.programme_filtering))
            ]
        if self.herkomst_filtering != []:
            data_to_predict = data_to_predict[
                (data_to_predict["Herkomst"].isin(self.herkomst_filtering))
            ]

        if len(data_to_predict) == 0:
            return None

        # Split the DataFrame into smaller chunks for parallel processing
        nr_CPU_cores = os.cpu_count()
        chunk_size = math.ceil(
            len(data_to_predict) / nr_CPU_cores
        )  # Make as much chunks as you have CPU cores

        chunks = [
            data_to_predict[i : i + chunk_size] for i in range(0, len(data_to_predict), chunk_size)
        ]

        print("Start parallel predicting...")
        # Use joblib.Parallel to parallelize the operation
        self.predicted_data = joblib.Parallel(n_jobs=nr_CPU_cores)(
            joblib.delayed(self.predict_with_sarima)(row)
            for chunk in chunks
            for _, row in chunk.iterrows()
        )

        # Create two new columns. 'Voorspelde vooraanmelders' is just predicted
        # and stored in predicted_data. This will be added with 'add_predicted_preregistrations()'.
        # These values will be used to predict the 'SARIMA_cumulative'.
        data_to_predict["SARIMA_individual"] = [x[0] for x in self.predicted_data]
        data_to_predict["Voorspelde vooraanmelders"] = np.nan

        # Add predicted preregistrations to the data_to_predict dataframe.
        if self.predict_week != 38:
            data_to_predict = self.helpermethods.add_predicted_preregistrations(
                data_to_predict, [x[1] for x in self.predicted_data]
            )

        # Predict the SARIMA_cumulative and add it to the dataframe.

        # data_to_predict = self.cumulative.predict_students_with_preapplicants(
        #    full_data, [x[1] for x in self.predicted_data], data_to_predict
        # )

        data_to_predict = self.cumulative.predict_students_with_preapplicants(
            full_data, [x[1] for x in self.predicted_data], data_to_predict
        )

        return data_to_predict

    # Predicts nr of students using both the individual and cumulative dataset
    def predict_with_sarima(self, row):
        print(
            f"Prediction for {row['Croho groepeernaam']}, {row['Herkomst']}, year: {self.predict_year}, week: {self.predict_week}"
        )

        sarima_individual = self.individual.predict_with_sarima(
            row, self.data_exog, already_printed=True
        )
        if self.predict_week == 38:
            return sarima_individual, []
        else:
            predicted_preregistration = self.cumulative.predict_with_sarima(
                row, already_printed=True
            )

        return sarima_individual, predicted_preregistration
