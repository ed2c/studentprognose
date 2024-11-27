import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def mapping(x):
    if x < 10:
        return "red"
    elif x < 50:
        return "blue"
    else:
        return "green"


def scatterplot(data, col, filter):
    mae_column = f"AVG_MAE_{col}"
    mape_column = f"AVG_MAPE_{col}"

    if col == "Ensemble_prediction":
        data = data[(data[mae_column] < 20) & (data[mape_column] < 2)]
    elif col == "SARIMA_individual":
        data = data[(data[mae_column] < 20) & (data[mape_column] < 2)]
    elif col == "SARIMA_cumulative":
        data = data[(data[mae_column] < 30) & (data[mape_column] < 2)]
    elif col == "Prognose_ratio":
        data = data[(data[mae_column] < 70) & (data[mape_column] < 3)]

    plt.figure(figsize=(10, 6))

    if filter == "examtype":
        colors = data["Examentype"].map({"Master": "blue", "Bachelor": "red"})
        label = "Master: Blue, Bachelor: Red"
    elif filter == "herkomst":
        colors = data["Herkomst"].map({"NL": "orange", "EER": "blue", "Niet-EER": "red"})
        label = "NL: Orange, EER: Blue, Niet-EER: Red"
    elif filter == "aantal_studenten":
        colors = data["Aantal_studenten"]
        label = "Aantal studenten"

    plt.scatter(
        data[mae_column],
        data[mape_column],
        c=colors,
        marker=".",
        alpha=0.7,
        label=label,
    )

    plt.xlabel(mae_column)
    plt.ylabel(mape_column)
    plt.title(f"Scatterplot of MAE vs MAPE ({col}) by {filter}")
    plt.legend()
    plt.grid(True)

    if filter == "aantal_studenten":
        plt.colorbar()

    plt.savefig(f"data/output/scatterplots/{col}_{filter}.png")


if __name__ == "__main__":
    data = pd.read_excel("data/output/evaluation_results.xlsx")

    cols = ["Ensemble_prediction", "SARIMA_individual", "SARIMA_cumulative", "Prognose_ratio"]
    filters = ["examtype", "herkomst", "aantal_studenten"]
    for col in cols:
        for filter in filters:
            scatterplot(data, col, filter)
