import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.load_data import load_configuration

year_to_evaluate = 2024


if __name__ == "__main__":
    configuration = load_configuration("configuration/configuration.json")

    data_total = pd.read_excel(configuration["paths"]["path_latest"])

    data = data_total[data_total["Collegejaar"] == year_to_evaluate]

    data["AVG_SARIMA_cumulative"] = None
    data["AVG_SARIMA_individual"] = None
    data["AVG_Ensemble_prediction"] = None
    data["AVG_MAPE_Ensemble_prediction"] = None

    for programme in data["Croho groepeernaam"].unique():
        for examtype in data[data["Croho groepeernaam"] == programme]["Examentype"].unique():
            for herkomst in data[
                (data["Croho groepeernaam"] == programme) & (data["Examentype"] == examtype)
            ]["Herkomst"].unique():
                data.loc[
                    (data["Croho groepeernaam"] == programme)
                    & (data["Examentype"] == examtype)
                    & (data["Herkomst"] == herkomst),
                    "AVG_MAPE_Ensemble_prediction",
                ] = data[
                    (data["Croho groepeernaam"] == programme)
                    & (data["Examentype"] == examtype)
                    & (data["Herkomst"] == herkomst)
                ][
                    "MAPE_Ensemble_prediction"
                ].mean()

                data.loc[
                    (data["Croho groepeernaam"] == programme)
                    & (data["Examentype"] == examtype)
                    & (data["Herkomst"] == herkomst),
                    "AVG_Ensemble_prediction",
                ] = data[
                    (data["Croho groepeernaam"] == programme)
                    & (data["Examentype"] == examtype)
                    & (data["Herkomst"] == herkomst)
                ][
                    "Ensemble_prediction"
                ].mean()

                data.loc[
                    (data["Croho groepeernaam"] == programme)
                    & (data["Examentype"] == examtype)
                    & (data["Herkomst"] == herkomst),
                    "AVG_SARIMA_cumulative",
                ] = data[
                    (data["Croho groepeernaam"] == programme)
                    & (data["Examentype"] == examtype)
                    & (data["Herkomst"] == herkomst)
                ][
                    "SARIMA_cumulative"
                ].mean()

                data.loc[
                    (data["Croho groepeernaam"] == programme)
                    & (data["Examentype"] == examtype)
                    & (data["Herkomst"] == herkomst),
                    "AVG_SARIMA_individual",
                ] = data[
                    (data["Croho groepeernaam"] == programme)
                    & (data["Examentype"] == examtype)
                    & (data["Herkomst"] == herkomst)
                ][
                    "SARIMA_individual"
                ].mean()

    # print(data[(data["Croho groepeernaam"] == "B Bedrijfskunde") & (data["Examentype"] == "Bachelor") & (data["Herkomst"] == "NL")]["AVG_MAPE_Ensemble_prediction"])
    data = data.drop_duplicates(
        subset=["Croho groepeernaam", "Examentype", "Herkomst"], ignore_index=True
    )
    data = data[
        [
            "Examentype",
            "Croho groepeernaam",
            "Herkomst",
            "Aantal_studenten",
            "AVG_SARIMA_cumulative",
            "AVG_SARIMA_individual",
            "AVG_Ensemble_prediction",
            "AVG_MAPE_Ensemble_prediction",
        ]
    ]
    data = data[data["Aantal_studenten"] > 10]
    data = data.sort_values(
        by=["AVG_MAPE_Ensemble_prediction", "Croho groepeernaam", "Examentype", "Herkomst"],
        ascending=[False, True, True, True],
        ignore_index=True,
    )

    CWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outfile = os.path.join(CWD, "data/output/evaluation_results.xlsx")
    data.to_excel(outfile, index=False)
