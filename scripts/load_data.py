import pandas as pd
import json
import os


def load_data(configuration):
    paths = configuration["paths"]

    data_individual = (
        pd.read_csv(paths["path_individual"], sep=";", skiprows=[1])
        if (paths["path_individual"] != "" and os.path.exists(paths["path_individual"]))
        else None
    )
    data_cumulative = (
        pd.read_csv(paths["path_cumulative"], sep=";", skiprows=[1], low_memory=True)
        if (paths["path_cumulative"] != "" and os.path.exists(paths["path_cumulative"]))
        else None
    )
    data_latest = (
        pd.read_excel(paths["path_latest"])
        if (paths["path_latest"] != "" and os.path.exists(paths["path_latest"]))
        else None
    )
    data_distances = (
        pd.read_excel(paths["path_distances"])
        if (paths["path_distances"] != "" and os.path.exists(paths["path_distances"]))
        else None
    )
    data_weighted_ensemble = (
        pd.read_excel(paths["path_weighted_ensemble"])
        if (
            paths["path_weighted_ensemble"] != ""
            and os.path.exists(paths["path_weighted_ensemble"])
        )
        else None
    )

    data_student_numbers_first_years = (
        pd.read_excel(paths["path_student_count_first-years"])
        if paths["path_student_count_first-years"] != ""
        else None
    )
    data_student_numbers_higher_years = (
        pd.read_excel(paths["path_student_count_higher-years"])
        if paths["path_student_count_higher-years"] != ""
        else None
    )
    data_student_numbers_volume = (
        pd.read_excel(paths["path_student_volume"]) if paths["path_student_volume"] != "" else None
    )

    data_october = pd.read_excel(paths["path_october"]) if paths["path_october"] != "" else None

    data_ratios = pd.read_excel(paths["path_ratios"]) if paths["path_ratios"] != "" else None

    if data_individual is not None:
        columns_i = configuration["columns"]["individual"]
        data_individual = data_individual.rename(
            columns={
                columns_i["Sleutel"]: "Sleutel",
                columns_i["Datum Verzoek Inschr"]: "Datum Verzoek Inschr",
                columns_i["Ingangsdatum"]: "Ingangsdatum",
                columns_i["Collegejaar"]: "Collegejaar",
                columns_i["Datum intrekking vooraanmelding"]: "Datum intrekking vooraanmelding",
                columns_i["Inschrijfstatus"]: "Inschrijfstatus",
                columns_i["Faculteit"]: "Faculteit",
                columns_i["Examentype"]: "Examentype",
                columns_i["Croho"]: "Croho",
                columns_i["Croho groepeernaam"]: "Croho groepeernaam",
                columns_i["Opleiding"]: "Opleiding",
                columns_i["Hoofdopleiding"]: "Hoofdopleiding",
                columns_i["Eerstejaars croho jaar"]: "Eerstejaars croho jaar",
                columns_i["Is eerstejaars croho opleiding"]: "Is eerstejaars croho opleiding",
                columns_i["Is hogerejaars"]: "Is hogerejaars",
                columns_i["BBC ontvangen"]: "BBC ontvangen",
                columns_i["Type vooropleiding"]: "Type vooropleiding",
                columns_i["Nationaliteit"]: "Nationaliteit",
                columns_i["EER"]: "EER",
                columns_i["Geslacht"]: "Geslacht",
                columns_i["Geverifieerd adres postcode"]: "Geverifieerd adres postcode",
                columns_i["Geverifieerd adres plaats"]: "Geverifieerd adres plaats",
                columns_i["Geverifieerd adres land"]: "Geverifieerd adres land",
                columns_i["Studieadres postcode"]: "Studieadres postcode",
                columns_i["Studieadres land"]: "Studieadres land",
                columns_i["School code eerste vooropleiding"]: "School code eerste vooropleiding",
                columns_i["School eerste vooropleiding"]: "School eerste vooropleiding",
                columns_i["Plaats code eerste vooropleiding"]: "Plaats code eerste vooropleiding",
                columns_i["Land code eerste vooropleiding"]: "Land code eerste vooropleiding",
                columns_i["Aantal studenten"]: "Aantal studenten",
            }
        )

    if data_cumulative is not None:
        columns_c = configuration["columns"]["cumulative"]
        data_cumulative = data_cumulative.rename(
            columns={
                columns_c["Korte naam instelling"]: "Korte naam instelling",
                columns_c["Collegejaar"]: "Collegejaar",
                columns_c["Weeknummer rapportage"]: "Weeknummer rapportage",
                columns_c["Weeknummer"]: "Weeknummer",
                columns_c["Faculteit"]: "Faculteit",
                columns_c["Type hoger onderwijs"]: "Type hoger onderwijs",
                columns_c["Groepeernaam Croho"]: "Groepeernaam Croho",
                columns_c["Naam Croho opleiding Nederlands"]: "Naam Croho opleiding Nederlands",
                columns_c["Croho"]: "Croho",
                columns_c["Herinschrijving"]: "Herinschrijving",
                columns_c["Hogerejaars"]: "Hogerejaars",
                columns_c["Herkomst"]: "Herkomst",
                columns_c["Gewogen vooraanmelders"]: "Gewogen vooraanmelders",
                columns_c["Ongewogen vooraanmelders"]: "Ongewogen vooraanmelders",
                columns_c[
                    "Aantal aanmelders met 1 aanmelding"
                ]: "Aantal aanmelders met 1 aanmelding",
                columns_c["Inschrijvingen"]: "Inschrijvingen",
            }
        )

    return (
        data_individual,
        data_cumulative,
        data_student_numbers_first_years,
        data_student_numbers_higher_years,
        data_student_numbers_volume,
        data_latest,
        data_distances,
        data_weighted_ensemble,
        data_october,
        data_ratios,
    )


def load_configuration(file_path):
    f = open(file_path)

    data = json.load(f)

    f.close()

    return data
