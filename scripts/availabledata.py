from scripts.helper import *
from scripts.datatotal import *

from abc import ABC, abstractmethod
import collections


class AvailableData(ABC):
    def __init__(self, configuration):
        self.numerus_fixus_list = configuration["numerus_fixus"]

        self.data_total = DataTotal()

        self.programme_filtering = []
        self.herkomst_filtering = []

    @abstractmethod
    def preprocess(self, programme_filtering, herkomst_filtering):
        return None

    @abstractmethod
    def predict_nr_of_students(self, predict_year, predict_week, student_year_prediction):
        pass

    def set_year_week(self, predict_year, predict_week, data):
        self.predict_year = predict_year
        self.predict_week = predict_week

        self.max_year = data["Collegejaar"].max()
        self.max_week = get_max_week(self.predict_year, self.max_year, data, "Collegejaar")

    def set_filtering(self, programme_filtering, herkomst_filtering):
        self.programme_filtering = programme_filtering
        self.herkomst_filtering = herkomst_filtering

    def get_data_to_predict(self, data, programme_filtering=[], herkomst_filtering=[]):
        predict_dict = {"Croho groepeernaam": [], "Herkomst": [], "Collegejaar": [], "Weeknummer": [], "Examentype": [], "Faculteit": []}

        all_programmes = data["Croho groepeernaam"].unique()
        if programme_filtering != []:
            all_programmes = list((collections.Counter(all_programmes) & collections.Counter(programme_filtering)).elements())

        all_herkomsts = data["Herkomst"].unique()
        if herkomst_filtering != []:
            all_herkomsts = list((collections.Counter(all_herkomsts) & collections.Counter(herkomst_filtering)).elements())

        for programme in np.sort(all_programmes):
            for herkomst in np.sort(all_herkomsts):
                predict_dict["Croho groepeernaam"].append(programme)
                predict_dict["Herkomst"].append(herkomst)

                predict_dict["Collegejaar"].append(self.predict_year)
                predict_dict["Weeknummer"].append(self.predict_week)

                sample_row = data[data["Croho groepeernaam"] == programme].head(1)
                predict_dict["Examentype"].append(sample_row["Examentype"].values[0])
                predict_dict["Faculteit"].append(sample_row["Faculteit"].values[0])

        return pd.DataFrame(predict_dict)