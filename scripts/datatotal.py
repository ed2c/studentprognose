from scripts.dataoption import *
from scripts.transform_data import *
from scripts.helper import *

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json as json

class DataTotal:
    def __init__(self):
        self.average_ratio_between = (2021, 2023)

    def initialize(self, data_latest, ensemble_weights, data_studentcount, configuration, CWD):
        self.data_latest = data_latest
        self.ensemble_weights = ensemble_weights
        self.data_studentcount = data_studentcount

        self.numerus_fixus_list = configuration["numerus_fixus"]

        self.CWD = CWD

    def append_predicted_data(self, data, predicted_data, student_year_prediction):
        sample = predicted_data[0]
        if not isinstance(sample, tuple):
            # Only individual data
            data["SARIMA_individual"] = predicted_data
            data["SARIMA_cumulative"] = np.nan
            data["Voorspelde vooraanmelders"] = np.nan
        elif isinstance(sample, tuple) and len(sample) == 2:
            # Only cumulative data
            data["SARIMA_individual"] = np.nan
            data["SARIMA_cumulative"] = [x[0] for x in predicted_data]
            data["Voorspelde vooraanmelders"] = np.nan

            predicted_preregistrations = [x[1] for x in predicted_data]
            data = self._add_predicted_preregistrations(data, predicted_preregistrations)
        elif isinstance(sample, tuple) and len(sample) == 3:
            # Both datasets
            data["SARIMA_individual"] = [x[0] for x in predicted_data]
            data["SARIMA_cumulative"] = [x[1] for x in predicted_data]
            data["Voorspelde vooraanmelders"] = np.nan

            predicted_preregistrations = [x[2] for x in predicted_data]
            data = self._add_predicted_preregistrations(data, predicted_preregistrations)

        if student_year_prediction == StudentYearPrediction.FIRST_YEARS:
            data = self._numerus_fixus_cap(data)

        return data[['Collegejaar', 'Faculteit', 'Examentype', 'Herkomst', 'Croho groepeernaam', 'Weeknummer',
                          'SARIMA_cumulative', 'SARIMA_individual', 'Voorspelde vooraanmelders']]

    def _add_predicted_preregistrations(self, data, predicted_preregistrations):
        dict = {"Collegejaar": [], "Faculteit": [], "Examentype": [], "Herkomst": [], "Croho groepeernaam": [],
                "Weeknummer": [], "SARIMA_cumulative": [], "SARIMA_individual": [], "Voorspelde vooraanmelders": []}
        
        index = 0
        for _, row in data.iterrows():
            current_predicted_preregistrations = predicted_preregistrations[index]
            
            current_week = increment_week(row["Weeknummer"])
            for current_prediction in current_predicted_preregistrations:
                dict["Collegejaar"].append(row["Collegejaar"])
                dict["Faculteit"].append(row["Faculteit"])
                dict["Examentype"].append(row["Examentype"])
                dict["Herkomst"].append(row["Herkomst"])
                dict["Croho groepeernaam"].append(row["Croho groepeernaam"])
                dict["Weeknummer"].append(current_week)
                dict["SARIMA_cumulative"].append(np.nan)
                dict["SARIMA_individual"].append(np.nan)
                dict["Voorspelde vooraanmelders"].append(current_prediction)

                current_week = increment_week(current_week)

            index += 1

        return pd.concat([data, pd.DataFrame(dict)], ignore_index=True)


    def _numerus_fixus_cap(self, data):
        for year in data["Collegejaar"].unique():
            for week in data["Weeknummer"].unique():
                for nf in self.numerus_fixus_list:
                    nf_data = data[(data["Collegejaar"] == year) & (data["Weeknummer"] == week) & (data["Croho groepeernaam"] == nf)]
                    if np.sum(nf_data['SARIMA_individual']) > self.numerus_fixus_list[nf]:
                        data.loc[(data["Collegejaar"] == year) & (data["Weeknummer"] == week) & (data["Croho groepeernaam"] == nf)
                                        & (data["Herkomst"] == "NL"), "SARIMA_individual"] = nf_data[nf_data["Herkomst"] == "NL"]["SARIMA_individual"] - (
                            np.sum(nf_data['SARIMA_individual']) - self.numerus_fixus_list[nf]
                        )

                    if np.sum(nf_data['SARIMA_cumulative']) > self.numerus_fixus_list[nf]:
                        data.loc[(data["Collegejaar"] == year) & (data["Weeknummer"] == week) & (data["Croho groepeernaam"] == nf)
                                        & (data["Herkomst"] == "NL"), "SARIMA_cumulative"] = nf_data[nf_data["Herkomst"] == "NL"]["SARIMA_cumulative"] - (
                            np.sum(nf_data['SARIMA_cumulative']) - self.numerus_fixus_list[nf]
                        )

        return data

    def prepare_data(self, data, data_cumulative=None):
        self.data = data

        prediction_df = pd.DataFrame(columns=['Collegejaar', 'Faculteit', 'Examentype', 'Herkomst', 'Croho groepeernaam', 'Weeknummer', 'SARIMA'])
        self.data = pd.concat([prediction_df, self.data])

        if data_cumulative is not None:
            self.data = self.data.merge(data_cumulative, on=['Croho groepeernaam', 'Collegejaar', 'Weeknummer', 'Herkomst'], how='left')
            self.data = self.data[['Croho groepeernaam', 'Collegejaar', 'Herkomst', 'Weeknummer', 'SARIMA_cumulative', 'SARIMA_individual', 'Voorspelde vooraanmelders']]

        if self.data_studentcount is not None:
            self.data = self.data.merge(self.data_studentcount, on=['Croho groepeernaam', 'Collegejaar', 'Herkomst'], how='left')

        if data_cumulative is not None:
            self.data = self.data.merge(data_cumulative, on=['Croho groepeernaam', 'Collegejaar', 'Herkomst', 'Weeknummer'], how='left')

        output_path = os.path.join(self.CWD, 'data', 'output', 'output_prelim.xlsx')
        self.data.to_excel(output_path, index=False)

        self.data['Faculteit'].replace({
                'LET': 'FdL',
                'SOW': 'FSW',
                'RU': 'FdM',
                'MAN': 'FdM',
                'NWI': 'FNWI',
                'MED': 'FMW',
                'FTR': 'FFTR',
                'JUR': 'FdR'
            }, inplace=True)
        
        output_path = os.path.join(self.CWD, 'data', 'output', 'output_prelim.xlsx')
        self.data.to_excel(output_path, index=False)

    def predict_with_ratio(self, data_cumulative):
        if self.data_studentcount is not None:
            data_vooraanmeldingen = data_cumulative[['Collegejaar', 'Weeknummer', 'Croho groepeernaam', 'Herkomst', 'Ongewogen vooraanmelders', 'Inschrijvingen']]
            data_vooraanmeldingen['Ongewogen vooraanmelders'] = data_vooraanmeldingen['Ongewogen vooraanmelders'].astype('float')
            data_vooraanmeldingen['Aanmeldingen'] = data_vooraanmeldingen['Ongewogen vooraanmelders'] + data_vooraanmeldingen['Inschrijvingen']
            data_vooraanmeldingen = data_vooraanmeldingen[(data_vooraanmeldingen['Collegejaar'] >= self.average_ratio_between[0]) & (data_vooraanmeldingen['Collegejaar'] <= self.average_ratio_between[1])]

            data_merged = pd.merge(left=data_vooraanmeldingen, right=self.data_studentcount, on=['Collegejaar', 'Croho groepeernaam', 'Herkomst'], how='left')
            data_merged['Average_Ratio'] = data_merged['Aanmeldingen'].divide(data_merged['Aantal_studenten'])
            data_merged['Average_Ratio'] = data_merged['Average_Ratio'].replace(np.inf, np.nan)

            Average_Ratios = data_merged[['Croho groepeernaam', 'Herkomst', 'Weeknummer', 'Average_Ratio']].groupby(['Croho groepeernaam', 'Herkomst', 'Weeknummer'], as_index=False).mean()
            
            self.data = self.data.rename(columns={'Aantal eerstejaarsopleiding': 'EOI_vorigjaar'})
            self.data['Ongewogen vooraanmelders'] = self.data['Ongewogen vooraanmelders'].astype('float')
            self.data['Aanmelding'] = self.data['Ongewogen vooraanmelders'] + self.data['Inschrijvingen']

            self.data['Ratio'] = (self.data['Aanmelding']).divide(self.data['Aantal_studenten'])
            self.data['Ratio'] = self.data['Ratio'].replace(np.inf, np.nan)

            self.data = pd.merge(left=self.data, right=Average_Ratios, on=['Croho groepeernaam', 'Herkomst', 'Weeknummer'])

            for i, row in self.data.iterrows():
                if row["Average_Ratio"] != 0:
                    self.data.at[i, "Prognose_ratio"] = row['Aanmelding'] / row['Average_Ratio']

            NFs = self.numerus_fixus_list
            for year in self.data["Collegejaar"].unique():
                for week in self.data["Weeknummer"].unique():
                    for nf in NFs:
                        nf_data = self.data[(self.data["Collegejaar"] == year) & (self.data["Weeknummer"] == week) &
                                                            (self.data["Croho groepeernaam"] == nf)]

                        if np.sum(nf_data['Prognose_ratio']) > NFs[nf]:
                            self.data.loc[(self.data["Collegejaar"] == year) & (self.data["Weeknummer"] == week) & (self.data["Croho groepeernaam"] == nf)
                                            & (self.data["Herkomst"] == "NL"), "Prognose_ratio"] = nf_data[nf_data["Herkomst"] == "NL"]["Prognose_ratio"] - (
                                np.sum(nf_data['Prognose_ratio']) - NFs[nf]
                            )

    def postprocess(self, postprocess_subset, predict_year, predict_week):
        self.postprocess_subset = postprocess_subset
        self.predict_year = predict_year
        self.predict_week = predict_week

        if self.data_latest is not None:
            self.data = replace_latest_data_old(self.data_latest, self.data)#, predict_year, predict_week)

        # Postprocess the total data, i.e. the forecasted and latest data
        self._create_ensemble_columns()
        self._create_error_columns()

    def _get_normal_ensemble(self, row):
        sarima_cumulative = convert_nan_to_zero(row["SARIMA_cumulative"])
        sarima_individual = convert_nan_to_zero(row["SARIMA_individual"])
        ensemble_prediction = None
        if row["Croho groepeernaam"] in ['B Geneeskunde', 'B Biomedische Wetenschappen', 'B Tandheelkunde']:
            ensemble_prediction = sarima_cumulative
        
        elif row["Weeknummer"] in range(17, 23 + 1) and row["Examentype"] == 'Master':
            ensemble_prediction = sarima_individual * 0.2 + sarima_cumulative * 0.8
        
        elif row["Weeknummer"] in range(30, 34 + 1):
            ensemble_prediction = sarima_individual * 0.6 + sarima_cumulative * 0.4
        
        elif row["Weeknummer"] in range(35, 37 + 1):
            ensemble_prediction = sarima_individual * 0.7 + sarima_cumulative * 0.3

        elif row["Weeknummer"] == 38:
            ensemble_prediction = sarima_individual

        else:
            ensemble_prediction = sarima_individual * 0.5 + sarima_cumulative * 0.5

        return ensemble_prediction

    def _create_ensemble_columns(self):
        self.data = self.data.sort_values(by=["Croho groepeernaam", "Herkomst", "Collegejaar", "Weeknummer"])
        self.data = self.data.reset_index(drop=True)

        if self.postprocess_subset == PostProcessSubset.ALL:
            self.data["Ensemble_prediction"] = np.nan
            self.data["Weighted_ensemble_predicition"] = -1.0
        for index, row in self.data.iterrows():
            if self.postprocess_subset == PostProcessSubset.NEW and (row["Collegejaar"] != self.predict_year or row["Weeknummer"] != self.predict_week):
                continue

            normal_ensemble = self._get_normal_ensemble(row)

            self.data.at[index, "Ensemble_prediction"] = normal_ensemble

            if self.ensemble_weights is not None:
                temp_weighted_data = self.ensemble_weights[self.ensemble_weights["Programme"] == row["Croho groepeernaam"]]
                temp_weighted_data = temp_weighted_data[temp_weighted_data["Herkomst"] == row["Herkomst"]]
                if len(temp_weighted_data) > 0:
                    temp_weighted_data = temp_weighted_data.iloc[0]

                    if temp_weighted_data["Average_ensemble_prediction"] != 1:
                        weighted_ensemble = convert_nan_to_zero(row["SARIMA_cumulative"]) * temp_weighted_data["SARIMA_cumulative"] + convert_nan_to_zero(row["SARIMA_individual"]) * temp_weighted_data["SARIMA_individual"] + convert_nan_to_zero(row["Prognose_ratio"]) * temp_weighted_data["Prognose_ratio"]
                        
                        if not np.isnan(weighted_ensemble):
                            self.data.at[index, "Weighted_ensemble_predicition"] = weighted_ensemble

        if self.postprocess_subset == PostProcessSubset.ALL:
            self.data["Average_ensemble_prediction"] = np.nan

        for index, row in self.data.iterrows():
            if self.postprocess_subset == PostProcessSubset.NEW and (row["Collegejaar"] != self.predict_year or row["Weeknummer"] != self.predict_week):
                continue

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
                if i >= 0 and self.data.at[i, "Croho groepeernaam"] == current_programme and self.data.at[i, "Herkomst"] == current_origin and \
                    self.data.at[i, "Weeknummer"] == offset_week and self.data.at[i, "Collegejaar"] == offset_year:
                    total += self.data.at[i, "Ensemble_prediction"]
                    nr_of_samples += 1
            if nr_of_samples == 0:
                nr_of_samples = 1
            average = total / nr_of_samples
            self.data.at[index, "Average_ensemble_prediction"] = average
            if row["Weighted_ensemble_predicition"] == -1.0:
                self.data.at[index, "Weighted_ensemble_predicition"] = average

    def _mean_absolute_error(self, row, key):
        return abs(row["Aantal_studenten"] - convert_nan_to_zero(row[key]))

    def _mean_absolute_percentage_error(self, row, key):
        return abs((row["Aantal_studenten"] - convert_nan_to_zero(row[key])) / row["Aantal_studenten"])

    def _create_error_columns(self):
        if self.postprocess_subset == PostProcessSubset.ALL:
            self.data["MAE weighted ensemble"] = np.nan
            self.data["MAE average ensemble"] = np.nan
            self.data["MAE ensemble"] = np.nan
            self.data["MAE ratio"] = np.nan
            self.data["MAE sarima cumulative"] = np.nan
            self.data["MAE sarima individual"] = np.nan

            self.data["MAPE weighted ensemble"] = np.nan
            self.data["MAPE average ensemble"] = np.nan
            self.data["MAPE ensemble"] = np.nan
            self.data["MAPE ratio"] = np.nan
            self.data["MAPE sarima cumulative"] = np.nan
            self.data["MAPE sarima individual"] = np.nan

        count = 0.0
        total_MAE_weighted_ensemble = 0.0
        total_MAE_average_ensemble = 0.0
        total_MAE_ensemble = 0.0
        total_MAE_ratio = 0.0
        total_MAE_cumulative = 0.0
        total_MAE_individual = 0.0
        for i, row in self.data.iterrows():
            if self.postprocess_subset == PostProcessSubset.NEW and (row["Collegejaar"] != self.predict_year or row["Weeknummer"] != self.predict_week):
                continue
            if row["Croho groepeernaam"] in self.numerus_fixus_list:
                continue

            self.data.at[i, "MAE weighted ensemble"] = self._mean_absolute_error(row, "Weighted_ensemble_predicition")
            self.data.at[i, "MAE average ensemble"] = self._mean_absolute_error(row, "Average_ensemble_prediction")
            self.data.at[i, "MAE ensemble"] = self._mean_absolute_error(row, "Ensemble_prediction")
            self.data.at[i, "MAE ratio"] = self._mean_absolute_error(row, "Prognose_ratio")
            self.data.at[i, "MAE sarima cumulative"] = self._mean_absolute_error(row, "SARIMA_cumulative")
            self.data.at[i, "MAE sarima individual"] = self._mean_absolute_error(row, "SARIMA_individual")

            self.data.at[i, "MAPE weighted ensemble"] = self._mean_absolute_percentage_error(row, "Weighted_ensemble_predicition")
            self.data.at[i, "MAPE average ensemble"] = self._mean_absolute_percentage_error(row, "Average_ensemble_prediction")
            self.data.at[i, "MAPE ensemble"] = self._mean_absolute_percentage_error(row, "Ensemble_prediction")
            self.data.at[i, "MAPE ratio"] = self._mean_absolute_percentage_error(row, "Prognose_ratio")
            self.data.at[i, "MAPE sarima cumulative"] = self._mean_absolute_percentage_error(row, "SARIMA_cumulative")
            self.data.at[i, "MAPE sarima individual"] = self._mean_absolute_percentage_error(row, "SARIMA_individual")

            if self.postprocess_subset == PostProcessSubset.NEW and not pd.isnull(row["Aantal_studenten"]):
                print(f"### Error metrics for {row['Croho groepeernaam']}, {row['Herkomst']}, year: {row['Collegejaar']}, week: {row['Weeknummer']} ###")
                print()
                print("MAE weighted ensemble:", self.data.at[i, "MAE weighted ensemble"])
                print("MAE average ensemble:", self.data.at[i, "MAE average ensemble"])
                print("MAE ensemble:", self.data.at[i, "MAE ensemble"])
                print("MAE ratio:", self.data.at[i, "MAE ratio"])
                print("MAE sarima cumulative:", self.data.at[i, "MAE sarima cumulative"])
                print("MAE sarima individual:", self.data.at[i, "MAE sarima individual"])
                print()
                print("MAPE weighted ensemble:", self.data.at[i, "MAPE weighted ensemble"])
                print("MAPE average ensemble:", self.data.at[i, "MAPE average ensemble"])
                print("MAPE ensemble:", self.data.at[i, "MAPE ensemble"])
                print("MAPE ratio:", self.data.at[i, "MAPE ratio"])
                print("MAPE sarima cumulative:", self.data.at[i, "MAPE sarima cumulative"])
                print("MAPE sarima individual:", self.data.at[i, "MAPE sarima individual"])
                print()

                total_MAE_weighted_ensemble += self.data.at[i, "MAE weighted ensemble"]
                total_MAE_average_ensemble += self.data.at[i, "MAE average ensemble"]
                total_MAE_ensemble += self.data.at[i, "MAE ensemble"]
                total_MAE_ratio += self.data.at[i, "MAE ratio"]
                total_MAE_cumulative += self.data.at[i, "MAE sarima cumulative"]
                total_MAE_individual += self.data.at[i, "MAE sarima individual"]
                count += 1.0

        if self.postprocess_subset == PostProcessSubset.NEW:
            if count == 0:
                count = 1.0

            print(f"### Average MAE ###")
            print()
            print("MAE weighted ensemble:", total_MAE_weighted_ensemble / count)
            print("MAE average ensemble:", total_MAE_average_ensemble / count)
            print("MAE ensemble:", total_MAE_ensemble / count)
            print("MAE ratio:", total_MAE_ratio / count)
            print("MAE sarima cumulative:", total_MAE_cumulative / count)
            print("MAE sarima individual:", total_MAE_individual / count)
            print()

    def save_output(self):
        self.data_latest = self.data

        # Adjust the file path to point to the correct location of the configuration file
        config_file_path = os.path.join(self.CWD, 'configuration', 'configuration.json')

        # Read the configuration file to get the filtered programme
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        filtered_programme = config["filtering"]["programme"][0]  # Assuming only one programme is filtered
        
        # Filter the data based on the filtered programme
        data_filtered = self.data[self.data["Croho groepeernaam"] == filtered_programme]

        # Define a custom sort key
        def custom_sort_key(weeknummer):
            if weeknummer >= 39:
                return weeknummer - 39
            else:
                return weeknummer + 52 - 39

        # Apply the custom sort key to the DataFrame
        data_filtered['SortKey'] = data_filtered['Weeknummer'].apply(custom_sort_key)

        # Sort the DataFrame based on the custom sort key
        df_sorted = data_filtered.sort_values(by='SortKey').drop(columns='SortKey')

        # Group by Weeknummer and sum the aantal_studenten
        grouped_data = df_sorted.groupby("Weeknummer").agg({"Aantal_studenten": "sum", "SARIMA_cumulative": "sum"}).reset_index()

        # Ensure grouped_data is sorted again by the custom order, though it should already be sorted correctly
        grouped_data['SortKey'] = grouped_data['Weeknummer'].apply(custom_sort_key)
        grouped_data = grouped_data.sort_values(by='SortKey').drop(columns='SortKey')

        # Extract necessary columns
        weeknummers = grouped_data["Weeknummer"]
        aantal_studenten = grouped_data["Aantal_studenten"]
        voorspelling = grouped_data["SARIMA_cumulative"]
        # Show only non-zero
        voorspelling_nonzero = voorspelling.replace(0, np.nan)

        # Plotting
        plt.figure(figsize=(12, 8))
        plt.plot(range(len(weeknummers)), aantal_studenten, label='Aantal Studenten', marker='o', linestyle='-')
        plt.plot(range(len(weeknummers)), voorspelling_nonzero, label='Voorspelling', marker='x', linestyle='--')

        plt.xlabel('Weeknummer')
        plt.ylabel('Aantal')
        plt.title(f'Aantal Studenten en Voorspelling per Week - {filtered_programme}')
        plt.legend()
        plt.grid(True)

        # Set x-ticks to the sorted week numbers
        plt.xticks(ticks=range(len(weeknummers)), labels=weeknummers)

        # Adjust y-axis limits for a more zoomed-out view
        ymin = min(min(aantal_studenten), min(voorspelling)) * 0.5
        ymax = max(max(aantal_studenten), max(voorspelling)) * 1.9
        plt.ylim(ymin, ymax)

        # Show the plot
        plt.show()

        # Save the plot as a PNG file
        plot_output_path = os.path.join(self.CWD, 'data', 'output', f'{filtered_programme}-{self.predict_year}-{self.predict_week}-plot.png')
        plt.savefig(plot_output_path)
        #plt.close()  # Close the plot to free memory

        # Save the data to Excel files
        input_output_path = os.path.join(self.CWD, 'data', 'input', 'totaal.xlsx')
        self.data.to_excel(input_output_path, index=False)

        output_output_path = os.path.join(self.CWD, 'data', 'output', 'output.xlsx')
        self.data.to_excel(output_output_path, index=False)