from scripts.availabledata import *
from scripts.helper import *
from scripts.transform_data import *

import numpy as np
from numpy import linalg as LA
import joblib
import os
import math
import gc
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

class Cumulative(AvailableData):
    def __init__(self, data_cumulative, data_studentcount, configuration, student_year_prediction):
        super().__init__(configuration)

        self.data_cumulative = data_cumulative
        self.data_studentcount = data_studentcount
        self.student_year_prediction = student_year_prediction

    def _preprocess(self, data):
        data = self._cast_string_to_float(data, "Ongewogen vooraanmelders")
        data = self._cast_string_to_float(data, "Gewogen vooraanmelders")
        data = self._cast_string_to_float(data, "Aantal aanmelders met 1 aanmelding")
        data = self._cast_string_to_float(data, "Inschrijvingen")
        
        data = data.groupby(['Collegejaar', 'Croho groepeernaam', 'Faculteit', 
                            'Examentype', 'Herkomst', 'Weeknummer']).sum(numeric_only=False).reset_index()
        return data[['Weeknummer', 'Collegejaar', 'Faculteit',
                    'Examentype', 'Herkomst', 'Croho groepeernaam',
                    'Gewogen vooraanmelders', 'Ongewogen vooraanmelders',
                    'Aantal aanmelders met 1 aanmelding', 'Inschrijvingen']]

    def preprocess(self):
        """
        Joins the main dataframe with the data with cumulative pre applications.

        Args:
            data (pd.DataFrame): The raw preapplication data to be joined and preprocessed.

        Returns:
            pd.DataFrame: A consolidated and preprocessed DataFrame containing preapplication information.

        This function performs the following tasks:
        1. Converts columns with comma-separated numbers to float64 data types.
        2. Groups the data by specific columns and sums non-numeric columns.
        3. Resets the index and selects relevant columns.
        4. Renames the 'Groepeernaam Croho' column to 'Croho groepeernaam'.
        """

        # Set cumulative data to small variable name for reading with more ease
        data = self.data_cumulative
        
        # Rename certain columns to match columns of data individual
        data = data.rename(columns={'Type hoger onderwijs': 'Examentype', 'Groepeernaam Croho': 'Croho groepeernaam'})

        self.first_years_data = data[data["Hogerejaars"] == "Nee"]

        self.higher_years_data = data[data["Hogerejaars"] == "Ja"]

        self.first_years_data = self._preprocess(self.first_years_data)
        self.higher_years_data = self._preprocess(self.higher_years_data)

        if self.student_year_prediction == StudentYearPrediction.FIRST_YEARS:
            self.data_cumulative = self.first_years_data
        elif self.student_year_prediction == StudentYearPrediction.HIGHER_YEARS:
            self.data_cumulative = self.higher_years_data
        elif self.student_year_prediction == StudentYearPrediction.VOLUME:
            self.data_cumulative = sum_volume_data_cumulative(self.first_years_data, self.higher_years_data)

        return self.data_cumulative

    def _cast_string_to_float(self, data, key):
        if pd.api.types.is_string_dtype(data[key].dtype):
            data[key] = data[key].str.replace('.', '')
            data[key] = data[key].str.replace(',', '.')
        data[key] = pd.to_numeric(data[key], errors='coerce')
        data[key] = data[key].astype('float64')

        return data

    def predict_nr_of_students(self, predict_year, predict_week, student_year_prediction):
        if student_year_prediction == StudentYearPrediction.FIRST_YEARS:
            self.data_cumulative = self.first_years_data.copy(deep=True)
        elif student_year_prediction == StudentYearPrediction.HIGHER_YEARS:
            self.data_cumulative = self.higher_years_data.copy(deep=True)

        self.set_year_week(predict_year, predict_week, self.data_cumulative)

        data_to_predict = self.data_cumulative[(self.data_cumulative["Collegejaar"] == self.predict_year) &
                                               (self.data_cumulative["Weeknummer"] == self.predict_week)]
        if self.programme_filtering != []:
            data_to_predict = data_to_predict[(data_to_predict["Croho groepeernaam"].isin(self.programme_filtering))]
        if self.herkomst_filtering != []:
            data_to_predict = data_to_predict[(data_to_predict["Herkomst"].isin(self.herkomst_filtering))]

        self.prepare_data()
        
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
        
    def prepare_data(self):
        if self.data_studentcount is not None:
            self.data_cumulative = self.data_cumulative.merge(self.data_studentcount, on=['Croho groepeernaam', 'Collegejaar', 'Herkomst'], how='left')
        self.data_cumulative['ts'] = self.data_cumulative['Gewogen vooraanmelders'] + self.data_cumulative['Inschrijvingen']

        self.data_cumulative = self.data_cumulative.drop_duplicates()

        self.data_cumulative['Faculteit'] = self.data_cumulative['Faculteit'].replace({'SOW': 'FSW', 'LET': 'FdL', 'FTR': 'FFTR',
                                                                            'NWI': 'FNWI', 'MAN': 'FdM', 'JUR': 'Fdr',
                                                                            'MED': 'FMW', 'RU': 'FdM'})

    # Predicts pre-registrations with sarima per programme/origin/week
    def predict_with_sarima(self, row, already_printed=False):
        programme = row['Croho groepeernaam']
        herkomst = row["Herkomst"]
        examtype = row["Examentype"]

        if not already_printed:
            print(f"Prediction for {programme}, {herkomst}, year: {self.predict_year}, week: {self.predict_week}")

        gc.collect()

        data = self.data_cumulative.copy()

        data = data.drop_duplicates()

        data = data[data["Collegejaar"] >= 2016]

        full_data = transform_data(data, 'ts')

        data = full_data.copy()

        data = data[data["Herkomst"] == herkomst]

        data = data[data["Collegejaar"] <= self.predict_year]

        data = data[data['Croho groepeernaam'] == programme]

        if int(self.predict_week) > 38:
            pred_len = 38 + 52 - int(self.predict_week)
        else:
            pred_len = 38 - int(self.predict_week)

        # Week 39 to 0
        data['39'] = 0
        full_data['39'] = 0

        def create_time_series(data: pd.DataFrame, pred_len: int) -> np.array:
            """
            Create a time series data array from a DataFrame for a given prediction length.

            Args:
                data (pd.DataFrame): The input DataFrame containing time series data.
                pred_len (int): The length of the time series to be created.

            Returns:
                np.ndarray: A NumPy array containing the time series data.
            """

            ts_data = data.loc[:, get_all_weeks_valid(data.columns)].values.flatten()
            ts_data = ts_data[:-pred_len]

            return np.array(ts_data)

        ts_data = create_time_series(data, pred_len)
        if self.predict_week == 38:
            prediction = self._predict_with_xgboost(full_data, programme, examtype, herkomst)
            return prediction

        try:
            model = sm.tsa.statespace.SARIMAX(ts_data, order=(1, 0, 1), seasonal_order=(1, 1, 1, 52))
            results = model.fit(disp=0)

            pred = results.forecast(steps=pred_len)

            index = str(increment_week(self.predict_week))

            full_data.loc[
            (full_data["Collegejaar"] == self.predict_year) & (full_data['Croho groepeernaam'] == programme) & (full_data["Herkomst"] == herkomst),
            index:'38'] = pred

            prediction = self._predict_with_xgboost(full_data, programme, examtype, herkomst)
            return prediction, pred
        except (LA.LinAlgError, IndexError, ValueError) as error:
            print(f"Cumulative sarima error on: {programme}, {herkomst}")
            print(error)
            return np.nan, [np.nan]

    # Predicts nr of students based on pre-registration using xgboost
    def _predict_with_xgboost(self, data, programme, examtype, herkomst):
        try:
            # Train/test split
            if programme not in self.numerus_fixus_list:
                train = data[(data["Collegejaar"] < self.predict_year)  & (data["Examentype"] == examtype) & (
                    ~data['Croho groepeernaam'].isin(self.numerus_fixus_list))]
            elif programme in self.numerus_fixus_list:
                train = data[(data["Collegejaar"] < self.predict_year) & (data['Croho groepeernaam'] == programme)]

            test = data[(data["Collegejaar"] == self.predict_year) & (data['Croho groepeernaam'] == programme) & (data["Herkomst"] == herkomst)]

            if self.data_studentcount is not None:
                train = train.merge(self.data_studentcount[['Croho groepeernaam', 'Collegejaar', 'Herkomst', 'Aantal_studenten']],
                                    on=['Croho groepeernaam', 'Collegejaar', 'Herkomst'])
            else:
                # Student count is required
                return np.nan

            train = train.drop_duplicates()

            X_train = train.drop(['Aantal_studenten'], axis=1)
            y_train = train.pop('Aantal_studenten')

            # Encode
            # Specify the numeric and categorical column names
            numeric_cols = ['Collegejaar'] + [str(x) for x in get_weeks_list(38)]
            categorical_cols = ['Examentype', 'Faculteit', 'Croho groepeernaam', 'Herkomst']

            # Create transformers for numeric and categorical columns
            numeric_transformer = "passthrough"  # No transformation for numeric columns
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')

            # Create the column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numeric', numeric_transformer, numeric_cols),
                    ('categorical', categorical_transformer, categorical_cols)
                ])

            # Apply the preprocessing to the training and test data
            X_train = preprocessor.fit_transform(X_train)
            test = preprocessor.transform(test)
            # Model
            model = XGBRegressor(learning_rate=0.25)

            model.fit(X_train, y_train)

            predictie = model.predict(test)

            return int(round(predictie[0], 0))
        except ValueError:
            print(f"Cumulative xgboost error on: {programme}, {herkomst}")
            return np.nan
