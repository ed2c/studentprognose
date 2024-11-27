import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.load_data import load_configuration

year_to_evaluate = 2024


if __name__ == "__main__":
    configuration = load_configuration("configuration/configuration.json")

    data_total = pd.read_excel(configuration["paths"]["path_latest"])

    data = data_total[data_total["Collegejaar"] == year_to_evaluate]

    data = data[data["MAE_Ensemble_prediction"] != data["Aantal_studenten"]]

    data.replace([np.inf, -np.inf], None, inplace=True)

    cols = [
        "AVG_MAE_Ensemble_prediction",
        "AVG_MAPE_Ensemble_prediction",
        "AVG_MAE_SARIMA_individual",
        "AVG_MAPE_SARIMA_individual",
        "AVG_MAE_SARIMA_cumulative",
        "AVG_MAPE_SARIMA_cumulative",
        "AVG_MAE_Prognose_ratio",
        "AVG_MAPE_Prognose_ratio",
    ]
    for col in cols:
        data[col] = None

    for programme in data["Croho groepeernaam"].unique():
        for examtype in data[data["Croho groepeernaam"] == programme]["Examentype"].unique():
            for herkomst in data[
                (data["Croho groepeernaam"] == programme) & (data["Examentype"] == examtype)
            ]["Herkomst"].unique():
                for col in cols:
                    data.loc[
                        (data["Croho groepeernaam"] == programme)
                        & (data["Examentype"] == examtype)
                        & (data["Herkomst"] == herkomst),
                        col,
                    ] = data[
                        (data["Croho groepeernaam"] == programme)
                        & (data["Examentype"] == examtype)
                        & (data["Herkomst"] == herkomst)
                    ][
                        col.replace("AVG_", "")
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
        ]
        + cols
    ]
    data = data.sort_values(
        by=["AVG_MAPE_Ensemble_prediction", "Croho groepeernaam", "Examentype", "Herkomst"],
        ascending=[False, True, True, True],
        ignore_index=True,
    )

    CWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outfile = os.path.join(CWD, "data/output/evaluation_results.xlsx")
    data.to_excel(outfile, index=False)
