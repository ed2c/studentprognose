from configparser import ConfigParser


def load_data_old():
    # Load file paths
    cfg = ConfigParser()
    cfg.read("paths.cfg")
    path_individual = cfg.get("Paths", "path_individual")
    path_cumulative = cfg.get("Paths", "path_cumulative")
    path_student_numbers = cfg.get("Paths", "path_student_numbers")
    path_latest = cfg.get("Paths", "path_latest")
    path_distances = cfg.get("Paths", "path_distances")
    path_weighted_ensemble = cfg.get("Paths", "path_weighted_ensemble")

    # Load files into dataframes
    data_individual = pd.read_csv(path_individual, sep=";", skiprows=[1])
    data_cumulative = pd.read_csv(path_cumulative, sep=";", skiprows=[1])
    data_student_numbers = pd.read_excel(path_student_numbers)
    data_latest = pd.read_excel(path_latest)
    data_distances = pd.read_excel(path_distances)
    data_weighted_ensemble = pd.read_excel(path_weighted_ensemble)

    data_latest = data_latest.rename(
        columns={
            "SARIMA_old": "SARIMA_individual",
            "SARIMA_new": "SARIMA_cumulative",
            "Prognose": "Prognose_ratio",
        }
    )

    return (
        data_individual,
        data_cumulative,
        data_student_numbers,
        data_latest,
        data_distances,
        data_weighted_ensemble,
    )
