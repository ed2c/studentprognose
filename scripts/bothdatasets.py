from scripts.availabledata import *
from scripts.individual import *
from scripts.cumulative import *
from scripts.datatotal import *


class BothDatasets(AvailableData):
    def __init__(self, data_individual, data_cumulative, data_distances, data_studentcount, configuration, student_year_prediction):
        super().__init__(configuration)

        self.individual = Individual(data_individual, data_distances, configuration)
        self.cumulative = Cumulative(data_cumulative, data_studentcount, configuration, student_year_prediction)
        self.student_year_prediction = student_year_prediction

    def preprocess(self):
        print("Preprocessing individual data...")
        self.individual.preprocess()
        print("Preprocessing cumulative data...")
        return self.cumulative.preprocess()

    def predict_nr_of_students(self, predict_year, predict_week, student_year_prediction):
        if student_year_prediction == StudentYearPrediction.FIRST_YEARS:
            self.individual.data_individual = self.individual.first_years_data.copy(deep=True)
            self.cumulative.data_cumulative = self.cumulative.first_years_data.copy(deep=True)
        elif student_year_prediction == StudentYearPrediction.HIGHER_YEARS:
            self.individual.data_individual = self.individual.higher_years_data.copy(deep=True)
            self.cumulative.data_cumulative = self.cumulative.higher_years_data.copy(deep=True)

        self.set_year_week(predict_year, predict_week, self.cumulative.data_cumulative)
        self.individual.set_year_week(predict_year, predict_week, self.individual.data_individual)
        self.cumulative.set_year_week(predict_year, predict_week, self.cumulative.data_cumulative)

        data_to_predict = self.cumulative.data_cumulative[(self.cumulative.data_cumulative["Collegejaar"] == self.predict_year) &
                                               (self.cumulative.data_cumulative["Weeknummer"] == self.predict_week)]
        if self.programme_filtering != []:
            data_to_predict = data_to_predict[(data_to_predict["Croho groepeernaam"].isin(self.programme_filtering))]
        if self.herkomst_filtering != []:
            data_to_predict = data_to_predict[(data_to_predict["Herkomst"].isin(self.herkomst_filtering))]
        
        if len(data_to_predict) == 0:
            return None, None
        
        self.individual.data_individual = self.individual.data_individual.merge(self.cumulative.data_cumulative, on=['Croho groepeernaam', 'Collegejaar', 'Faculteit', 'Examentype', 'Weeknummer', 'Herkomst'], how='left')

        # Call predict_applicant first
        print("Predicting preapplicants...")
        predicties = self.individual.predict_applicant(self.individual.data_individual, self.cumulative.data_cumulative)
        self.individual.data_individual.loc[(self.individual.data_individual["Collegejaar"] == self.predict_year) & (self.individual.data_individual["Weeknummer"].isin(get_weeks_list(self.predict_week))), 'Inschrijvingen_predictie'] = predicties

        # Transform individual data to be used for sarima
        self.individual.transform_data_individual()
        
        temp_data_individual = self.individual.data_individual.copy(deep=True)
        temp_data_individual["Weeknummer"] = self.individual.data_individual["Weeknummer"].astype(int)

        # Create exogenous data for sarima with the individual dataset
        self.data_exog = temp_data_individual.merge(self.cumulative.data_cumulative,
                                                          on=['Croho groepeernaam', 'Collegejaar', 'Examentype',
                                                              'Faculteit', 'Weeknummer', 'Herkomst'], how='left')

        self.individual.data_individual = transform_data(self.individual.data_individual, 'Cumulative_sum_within_year')

        self.cumulative.prepare_data()
        
        # Split the DataFrame into smaller chunks for parallel processing
        nr_CPU_cores = os.cpu_count()
        chunk_size = math.ceil(len(data_to_predict) / nr_CPU_cores) # Make as much chunks as you have CPU cores

        chunks = [data_to_predict[i:i + chunk_size] for i in range(0, len(data_to_predict), chunk_size)]

        print("Start parallel predicting...")
        # Use joblib.Parallel to parallelize the operation
        self.predicted_data = joblib.Parallel(n_jobs=nr_CPU_cores)(
            joblib.delayed(self.predict_with_sarima)(row) for chunk in chunks for _, row in chunk.iterrows()
        )

        return data_to_predict, self.predicted_data

    # Predicts nr of students using both the individual and cumulative dataset
    def predict_with_sarima(self, row):
        print(f"Prediction for {row['Croho groepeernaam']}, {row['Herkomst']}, year: {self.predict_year}, week: {self.predict_week}")

        sarima_individual = self.individual.predict_with_sarima(row, self.data_exog, already_printed=True)
        sarima_cumulative, predicted_preregistration = self.cumulative.predict_with_sarima(row, already_printed=True)

        return sarima_individual, sarima_cumulative, predicted_preregistration
