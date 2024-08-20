import pandas as pd
import numpy as np
import sys
import collections


def subset(programme, herkomst):
    data_individual = pd.read_csv(
        "\\\\ru.nl\\wrkgrp\\TeamIR\\Man_info\\Student Analytics\\Prognosemodel RU\\Syntax\\Python\\studentprognose\\data\\input\\vooraanmeldingen_individueel.csv",
        sep=";",
        skiprows=[1],
    )
    data_cumulative = pd.read_csv(
        "\\\\ru.nl\\wrkgrp\\TeamIR\\Man_info\\Student Analytics\\Prognosemodel RU\\Syntax\\Python\\studentprognose\\data\\input\\vooraanmeldingen_cumulatief.csv",
        sep=";",
        skiprows=[1],
    )

    i = data_individual["Croho groepeernaam"].unique()
    c = data_cumulative["Groepeernaam Croho"].unique()

    programmes = list(
        (
            collections.Counter(data_individual["Croho groepeernaam"].unique())
            & collections.Counter(data_cumulative["Groepeernaam Croho"].unique())
        ).elements()
    )
    print("All programmes:", programmes)

    print(len(i))
    print(len(c))
    print(len(programmes))

    print("\nIndividual:\n")

    i2 = []
    for x in i:
        if x not in c:
            print(x)
            print(
                np.sort(
                    data_individual[data_individual["Croho groepeernaam"] == x][
                        "Collegejaar"
                    ].unique()
                )
            )

    print("\nCumulative:\n")

    c2 = []
    for x in c:
        if x not in i:
            print(x)
            print(
                np.sort(
                    data_cumulative[data_cumulative["Groepeernaam Croho"] == x][
                        "Collegejaar"
                    ].unique()
                )
            )

    def filter_herkomst(data, herkomst):
        if herkomst == "NL":
            return data[(data["Nationaliteit"] == "Nederlandse")]
        elif herkomst == "EER":
            return data[(data["Nationaliteit"] != "Nederlandse") & (data["EER"] == "J")]
        elif herkomst == "Niet-EER":
            return data[(data["Nationaliteit"] != "Nederlandse") & (data["EER"] != "J")]

    data_individual = data_individual[(data_individual["Croho groepeernaam"] == programme)]
    data_individual = filter_herkomst(data_individual, herkomst)

    data_cumulative = data_cumulative[
        (data_cumulative["Groepeernaam Croho"] == programme)
        & (data_cumulative["Herkomst"] == herkomst)
    ]

    data_individual.to_csv("vooraanmeldingen_individueel.csv", sep=";", index=False)
    data_cumulative.to_csv("vooraanmeldingen_cumulatief.csv", sep=";", index=False)


if __name__ == "__main__":
    programme = "M Computing Science"
    herkomst = "NL"

    if len(sys.argv) == 3:
        programme = sys.argv[1]
        herkomst = sys.argv[2]
    else:
        print("Not enough arguments given")

    print(f"Programme: {programme}, herkomst: {herkomst}")

    subset(programme, herkomst)
