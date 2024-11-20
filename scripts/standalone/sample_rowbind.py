import pandas as pd
from load_data import load_configuration

if __name__ == "__main__":
    configuration = load_configuration("configuration/configuration.json")

    data_cumulative = pd.read_csv(
        configuration["paths"]["path_cumulative"], sep=";", skiprows=[1], low_memory=True
    )
    data_student_numbers_first_years = pd.read_excel(
        configuration["paths"]["path_student_count_first-years"]
    )

    data_cumulative = data_cumulative[["Groepeernaam Croho", "Croho"]]
    data_cumulative.rename(columns={"Groepeernaam Croho": "Croho groepeernaam"}, inplace=True)
    data_cumulative = data_cumulative.drop_duplicates(subset="Croho groepeernaam")

    data = data_student_numbers_first_years.merge(
        data_cumulative, on="Croho groepeernaam", how="left"
    )
    data.drop(columns=["Croho groepeernaam"], inplace=True)
    data.rename(columns={"Croho": "Croho groepeernaam"}, inplace=True)
    data.drop_duplicates(inplace=True)

    # data_cumulative['Korte naam instelling'] = '21PM'
    # data_cumulative['Faculteit'] = 'FACULTEIT'
    # data_cumulative['Groepeernaam Croho'] = data_cumulative['Croho']
    # data_cumulative['Naam Croho opleiding Nederlands'] = data_cumulative['Croho']
    # data_cumulative[['Aantal aanmelders met 1 aanmelding', 'Inschrijvingen']] = None

    data.to_excel(
        "C:/Users/svenm/Documents/IR/studentprognose/data/input/studielink/example_studentcount_rowbinded.xlsx"
    )
