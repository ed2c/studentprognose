import pandas as pd
import numpy as np

data = pd.read_excel(
    "//ru.nl/wrkgrp/TeamIR/Man_info/Student Analytics/Prognosemodel RU/Syntax/Python/studentprognose/data/output/output.xlsx"
)

data = data.rename(
    columns={
        "MAE weighted ensemble": "MAE_Weighted_ensemble_prediction",
        "MAE average ensemble": "MAE_Average_ensemble_prediction",
        "MAE ensemble": "MAE_Ensemble_prediction",
        "MAE ratio": "MAE_Prognose_ratio",
        "MAE sarima cumulative": "MAE_SARIMA_cumulative",
        "MAE sarima individual": "MAE_SARIMA_individual",
        "MAPE weighted ensemble": "MAPE_Weighted_ensemble_prediction",
        "MAPE average ensemble": "MAPE_Average_ensemble_prediction",
        "MAPE ensemble": "MAPE_Ensemble_prediction",
        "MAPE ratio": "MAPE_Prognose_ratio",
        "MAPE sarima cumulative": "MAPE_SARIMA_cumulative",
        "MAPE sarima individual": "MAPE_SARIMA_individual",
    }
)

percentage_range = range(10, 150, 10)


def calculate_weight_distribution(metrics, percentage, second_percentage):
    # print(f"Metrics {metrics}")
    sorted(metrics, key=lambda x: x[1])
    # print(f"Metrics {metrics}")
    result = [(m[0], 0.0) for m in metrics]

    first_threshold = 1.0 + (float(percentage) / 100)
    second_threshold = 1.0 + (float(second_percentage) / 100)

    # Check if first method is best
    if metrics[1][1] > metrics[0][1] * first_threshold:
        result[0] = (result[0][0], 1.0)
        return result

    # Check if second method is still better than third
    elif metrics[2][1] > metrics[1][1] * second_threshold:
        m1 = metrics[0][1]
        m2 = metrics[1][1]
        total = m1 + m2
        new_total = total / m1 + total / m2

        result[0] = (result[0][0], (total / m1) / new_total)
        result[1] = (result[1][0], (total / m2) / new_total)
        return result

    # Combine all
    else:
        total = sum([m[1] for m in metrics])
        new_total = sum([total / m[1] for m in metrics])
        result = [(m[0], (total / m[1]) / new_total) for m in metrics]
        return result


def get_metric_weight_distribution(
    data, programme, herkomst, percentage, second_percentage, methods
):
    # print(programme, herkomst)
    data = data[data["Croho groepeernaam"] == programme]
    data = data[data["Herkomst"] == herkomst]

    MAE = [(method, np.nanmean(data[f"MAE_{method}"])) for method in methods]
    MAE = calculate_weight_distribution(MAE, percentage, second_percentage)
    # print(f"MAE: {MAE}")

    MAPE = [(method, np.nanmean(data[f"MAPE_{method}"]) / len(data) * 100) for method in methods]
    MAPE = calculate_weight_distribution(MAPE, percentage, second_percentage)
    # print(f"MAPE: {MAPE}")

    return MAE, MAPE


weight_distribution = {
    "Programme": [],
    "Herkomst": [],
    "Percentage": [],
    "Second percentage": [],
}

methods = ["SARIMA_cumulative", "SARIMA_individual", "Prognose_ratio"]

MAEs = []
MAPEs = []

print("Calculating weight distribution...")
for percentage in percentage_range:
    for second_percentage in percentage_range:
        for programme in data["Croho groepeernaam"].unique():
            for herkomst in data["Herkomst"].unique():
                weight_distribution["Programme"].append(programme)
                weight_distribution["Herkomst"].append(herkomst)
                weight_distribution["Percentage"].append(percentage)
                weight_distribution["Second percentage"].append(second_percentage)

                MAE, MAPE = get_metric_weight_distribution(
                    data, programme, herkomst, percentage, second_percentage, methods
                )

                MAEs.append(MAE)
                MAPEs.append(MAPE)

for m in methods:
    weight_distribution[f"MAE_{m}"] = [dict(MAE)[m] for MAE in MAEs]
    weight_distribution[f"MAPE_{m}"] = [dict(MAPE)[m] for MAPE in MAPEs]

weight_data = pd.DataFrame(weight_distribution)
weight_data.to_excel("configuration/weight_distribution.xlsx", index=False)


def mean_absolute_error(true, prediction):
    return abs(true - prediction)


def mean_absolute_percentage_error(true, prediction):
    return abs((true - prediction) / true)


print("Calculating error rates...")
error_rates = {
    "Percentage": [],
    "Second percentage": [],
    "MAE": [],
    "MAPE": [],
    "Herkomst": [],
    "MAE of AE": [],
    "MAPE of AE": [],
}
for herkomst in ["EER", "NL", "Niet-EER"]:
    for percentage in percentage_range:
        for second_percentage in percentage_range:
            error_rates["Percentage"].append(percentage)
            error_rates["Second percentage"].append(second_percentage)

            MAE_total = 0.0
            MAPE_total = 0.0

            MAE_total_AE = 0.0
            MAPE_total_AE = 0.0

            filtered_weight_data = weight_data[weight_data["Percentage"] == percentage]
            filtered_weight_data = filtered_weight_data[
                filtered_weight_data["Second percentage"] == second_percentage
            ]
            filtered_weight_data = filtered_weight_data[
                filtered_weight_data["Herkomst"] == herkomst
            ]
            for i, row_weight in filtered_weight_data.iterrows():
                filtered_data = data[data["Croho groepeernaam"] == row_weight["Programme"]]
                filtered_data = filtered_data[filtered_data["Herkomst"] == row_weight["Herkomst"]]
                filtered_data = filtered_data[filtered_data["Collegejaar"] >= 2021]

                use_average_ensemble = False
                if sum([row_weight[f"MAE_{method}"] for method in methods]) != 1:
                    use_average_ensemble = True

                MAE_subtotal = 0.0
                MAPE_subtotal = 0.0

                MAE_subtotal_AE = 0.0
                MAPE_subtotal_AE = 0.0
                for j, row in filtered_data.iterrows():
                    if use_average_ensemble:
                        if not np.isnan(row["MAE_Average_ensemble_prediction"]):
                            MAE_subtotal += row["MAE_Average_ensemble_prediction"]
                        if not np.isnan(row["MAPE_Average_ensemble_prediction"]):
                            MAPE_total += row["MAPE_Average_ensemble_prediction"]
                    else:
                        if all(
                            [(not np.isnan(row[method])) for method in methods]
                        ) and not np.isnan(row["Aantal_studenten"]):
                            new_ensemble_prediction = sum(
                                [(row[method] * row_weight[f"MAE_{method}"]) for method in methods]
                            )
                            MAE_subtotal += mean_absolute_error(
                                row["Aantal_studenten"], new_ensemble_prediction
                            )
                            MAPE_subtotal += mean_absolute_percentage_error(
                                row["Aantal_studenten"], new_ensemble_prediction
                            )

                    if not np.isnan(row["MAE_Average_ensemble_prediction"]):
                        MAE_subtotal_AE += row["MAE_Average_ensemble_prediction"]
                    if not np.isnan(row["MAPE_Average_ensemble_prediction"]):
                        MAPE_subtotal_AE += row["MAPE_Average_ensemble_prediction"]

                count = len(filtered_data)
                if count == 0:
                    count = 1

                MAE_total += MAE_subtotal / count
                MAPE_total += MAPE_subtotal / count * 100

                MAE_total_AE += MAE_subtotal_AE / count
                MAPE_total_AE += MAPE_subtotal_AE / count * 100

            error_rates["MAE"].append(MAE_total / len(filtered_weight_data))
            error_rates["MAPE"].append(MAPE_total / len(filtered_weight_data))
            error_rates["Herkomst"].append(herkomst)
            error_rates["MAE of AE"].append(MAE_total_AE / len(filtered_weight_data))
            error_rates["MAPE of AE"].append(MAPE_total_AE / len(filtered_weight_data))

error_rates = pd.DataFrame(error_rates)
error_rates.to_excel("configuration/error_rates.xlsx", index=False)

percentage_per_herkomst = {"EER": (), "NL": (), "Niet-EER": ()}
for herkomst in percentage_per_herkomst.keys():
    temp_data = error_rates[error_rates["Herkomst"] == herkomst]
    temp_data = temp_data[temp_data["MAE"] == temp_data["MAE"].min()]
    percentage_per_herkomst[herkomst] = (
        float(temp_data["Percentage"].iloc[0]),
        float(temp_data["Second percentage"].iloc[0]),
    )

print("Calculating ensemble weights...")
ensemble_weights = {
    "Programme": [],
    "Herkomst": [],
    "SARIMA_cumulative": [],
    "SARIMA_individual": [],
    "Prognose_ratio": [],
    "Average_ensemble_prediction": [],
}
for herkomst in percentage_per_herkomst.keys():
    temp_data = weight_data[weight_data["Herkomst"] == herkomst]
    temp_data = temp_data[temp_data["Percentage"] == percentage_per_herkomst[herkomst][0]]
    temp_data = temp_data[temp_data["Second percentage"] == percentage_per_herkomst[herkomst][1]]

    for i, row in temp_data.iterrows():
        ensemble_weights["Programme"].append(row["Programme"])
        ensemble_weights["Herkomst"].append(row["Herkomst"])

        average_ensemble_weight = 0.0

        MAEs = [row[f"MAE_{method}"] for method in methods]
        sum_MAE = sum(MAEs)

        if sum_MAE != 1 or sum_MAE == 0:
            average_ensemble_weight = 1.0
            MAEs = [0.0] * len(methods)

        for m in range(len(methods)):
            ensemble_weights[methods[m]].append(MAEs[m])

        ensemble_weights["Average_ensemble_prediction"].append(average_ensemble_weight)

ensemble_weights = pd.DataFrame(ensemble_weights)
ensemble_weights.to_excel("configuration/ensemble_weights.xlsx", index=False)
