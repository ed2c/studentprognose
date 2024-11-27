import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.load_data import load_configuration


def map_examtype(exam_code):
    if "Bachelor" in exam_code:
        return "Bachelor"
    elif "Master" in exam_code:
        return "Master"
    else:
        return "Pre-master"


def calculate_student_count(data, volume):
    dict = {
        "Collegejaar": [],
        "Croho groepeernaam": [],
        "Herkomst": [],
        "Aantal_studenten": [],
        "Examentype": [],
    }

    programme_key = "Groepeernaam Croho"
    herkomst_key = "EER-NL-nietEER"
    year_key = "Collegejaar"
    examtype_key = "Examentype code"

    for programme in data[programme_key].unique():
        for herkomst in data[herkomst_key].unique():
            years = data[year_key].unique()
            years = years[~np.isnan(years)]
            years = np.sort(years)
            for year in years:
                filtered_data = data[
                    (data[year_key] == year)
                    & (data[programme_key] == programme)
                    & (data[herkomst_key] == herkomst)
                ]

                if not volume:
                    filtered_data = filtered_data[
                        (filtered_data[examtype_key] == "Pre-master")
                        | (filtered_data["Aantal eerstejaars croho"] == 1)
                    ]
                    filtered_data = filtered_data[
                        (filtered_data[examtype_key] == "Bachelor eerstejaars")
                        | (filtered_data[examtype_key] == "Master")
                        | (filtered_data[examtype_key] == "Pre-master")
                    ]
                if volume:
                    filtered_data = filtered_data[
                        (filtered_data[examtype_key] == "Bachelor eerstejaars")
                        | (filtered_data[examtype_key] == "Master")
                        | (filtered_data[examtype_key] == "Pre-master")
                        | (filtered_data[examtype_key] == "Bachelor hogerejaars")
                    ]

                filtered_data = filtered_data[
                    ~filtered_data[examtype_key].str.contains("Master post initieel")
                ]

                # filtered_data[filtered_data[examtype_key].str.contains("Bachelor")].loc[examtype_key] = "Bachelor"

                filtered_data.loc[
                    filtered_data[examtype_key].str.contains("Bachelor"), examtype_key
                ] = "Bachelor"

                for examtype in filtered_data[examtype_key].unique():
                    examtype_filtered_data = filtered_data[filtered_data[examtype_key] == examtype]
                    student_count = np.sum(examtype_filtered_data["Aantal Hoofdinschrijvingen"])

                    if student_count > 0:
                        dict["Collegejaar"].append(year)
                        dict["Croho groepeernaam"].append(programme)
                        dict["Herkomst"].append(herkomst)
                        dict["Aantal_studenten"].append(student_count)
                        dict["Examentype"].append(map_examtype(examtype))

    dict = pd.DataFrame(dict)
    return dict


if __name__ == "__main__":
    CWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    configuration = load_configuration("configuration/configuration.json")
    data = pd.read_excel(configuration["paths"]["path_october"])

    # First-years student count including Pre-master
    output_path = os.path.join(CWD, "data", "input", "student_count_first-years.xlsx")
    dict_count = calculate_student_count(data, False)
    result_dict_count = dict_count[dict_count["Aantal_studenten"] > 0]
    result_dict_count.to_excel(output_path, index=False)

    # Volume student count including Pre-master
    output_path = os.path.join(CWD, "data", "input", "student_volume.xlsx")
    dict_volume = calculate_student_count(data, True)
    result_dict_volume = dict_volume[dict_volume["Aantal_studenten"] > 0]
    result_dict_volume.to_excel(output_path, index=False)

    # Higher-years student count excluding Pre-master
    dict_higher_years = dict_volume.copy()

    for idx, row in result_dict_volume.iterrows():
        match_row = result_dict_count[
            (result_dict_count["Collegejaar"] == row["Collegejaar"])
            & (result_dict_count["Croho groepeernaam"] == row["Croho groepeernaam"])
            & (result_dict_count["Herkomst"] == row["Herkomst"])
            & (result_dict_count["Examentype"] == row["Examentype"])
        ]
        if not match_row.empty:
            dict_higher_years.at[idx, "Aantal_studenten"] = (
                row["Aantal_studenten"] - match_row["Aantal_studenten"].values[0]
            )

    result_dict_higher_years = dict_higher_years[dict_higher_years["Aantal_studenten"] > 0]
    output_path = os.path.join(CWD, "data", "input", "student_count_higher-years.xlsx")
    result_dict_higher_years.to_excel(output_path, index=False)
