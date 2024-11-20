import os
import pandas as pd

directory = "C:/Users/svenm/Documents/IR/studentprognose/data/input/studielink"


if __name__ == "__main__":
    dataframes = []

    for file in os.listdir(directory):
        if file != "rowbinded.csv":
            weeknummer = file.replace("telbestandY2024W", "").replace(".csv", "")

            data = pd.read_csv(directory + "/" + file, sep=";", low_memory=False)
            data["Weeknummer"] = weeknummer

            dataframes.append(data)

    data = pd.concat(dataframes)

    data["Gewogen vooraanmelders"] = data["meercode_V"] / data["Aantal"]
    data.rename(
        columns={
            "Brincode": "Korte naam instelling",
            "Studiejaar": "Collegejaar",
            "Type_HO": "Type hoger onderwijs",
            "Isatcode": "Croho",
            "Aantal": "Ongewogen vooraanmelders",
        },
        inplace=True,
    )

    data = data[
        [
            "Korte naam instelling",
            "Croho",
            "Type hoger onderwijs",
            "Collegejaar",
            "Herkomst",
            "Hogerejaars",
            "Herinschrijving",
            "Ongewogen vooraanmelders",
            "Weeknummer",
            "Gewogen vooraanmelders",
        ]
    ]
    data[
        [
            "Weeknummer rapportage",
            "Faculteit",
            "Groepeernaam Croho",
            "Naam Croho opleiding Nederlands",
            "Aantal aanmelders met 1 aanmelding",
            "Inschrijvingen",
        ]
    ] = None

    data = data[
        [
            "Korte naam instelling",
            "Collegejaar",
            "Weeknummer rapportage",
            "Weeknummer",
            "Faculteit",
            "Type hoger onderwijs",
            "Groepeernaam Croho",
            "Naam Croho opleiding Nederlands",
            "Croho",
            "Herinschrijving",
            "Hogerejaars",
            "Herkomst",
            "Gewogen vooraanmelders",
            "Ongewogen vooraanmelders",
            "Aantal aanmelders met 1 aanmelding",
            "Inschrijvingen",
        ]
    ]

    data["Weeknummer rapportage"] = data["Weeknummer"]
    data["Faculteit"] = "FACULTEIT"
    data["Groepeernaam Croho"] = data["Croho"]
    data["Naam Croho opleiding Nederlands"] = data["Croho"]

    data["Type hoger onderwijs"] = data["Type hoger onderwijs"].replace(
        {"P": "Bachelor", "B": "Bachelor", "M": "Master"}
    )
    data["Herinschrijving"] = data["Herinschrijving"].replace({"J": "Ja", "N": "Nee"})
    data["Hogerejaars"] = data["Hogerejaars"].replace({"J": "Ja", "N": "Nee"})
    data["Herkomst"] = data["Herkomst"].replace({"N": "NL", "E": "EER", "R": "Niet-EER"})

    data.to_csv(
        "C:/Users/svenm/Documents/IR/studentprognose/data/input/studielink/rowbinded.csv",
        sep=";",
        index=False,
    )
