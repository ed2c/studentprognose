import unittest
import subprocess
import pandas as pd


class TestSarima(unittest.TestCase):
    def initialise(self):
        command = [
            "python",
            "main.py",
            "-w",
            "12",
            "-y",
            "2024",
            "-f",
            "configuration/filtering/test.json",
        ]

        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True)

        # Check that the command was successful
        self.assertEqual(result.returncode, 0, "The script did not run successfully")

        self.data = pd.read_excel("data/output/output_prelim.xlsx")

    def test_bedrijfskunde(self):
        data = self.data

        c_bedrijfskunde_nl_value = (
            data[
                (data["Croho groepeernaam"] == "B Bedrijfskunde")
                & (data["Herkomst"] == "NL")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_cumulative"].values[0]
        c_bedrijfskunde_eer_value = (
            data[
                (data["Croho groepeernaam"] == "B Bedrijfskunde")
                & (data["Herkomst"] == "EER")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_cumulative"].values[0]
        i_bedrijfskunde_nl_value = (
            data[
                (data["Croho groepeernaam"] == "B Bedrijfskunde")
                & (data["Herkomst"] == "NL")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_individual"].values[0]
        i_bedrijfskunde_eer_value = (
            data[
                (data["Croho groepeernaam"] == "B Bedrijfskunde")
                & (data["Herkomst"] == "EER")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_individual"].values[0]

        self.assertIn(
            c_bedrijfskunde_nl_value,
            [284, 324],
            "NL B Bedrijfskunde value (cumulative) is predicted wrongly ("
            + str(c_bedrijfskunde_nl_value)
            + " vs. [284, 324])",
        )
        self.assertEqual(
            c_bedrijfskunde_eer_value,
            42,
            "EER B Bedrijfskunde value (cumulative) is predicted wrongly ("
            + str(c_bedrijfskunde_eer_value)
            + " vs. 42)",
        )
        self.assertTrue(
            385 <= i_bedrijfskunde_nl_value <= 395,
            "NL B Bedrijfskunde value (individual) is predicted wrongly ("
            + str(i_bedrijfskunde_nl_value)
            + " vs. [385,395])",
        )
        self.assertTrue(
            25 <= i_bedrijfskunde_eer_value <= 35,
            "EER B Bedrijfskunde value (individual)  is predicted wrongly ("
            + str(i_bedrijfskunde_eer_value)
            + " vs. [25,35])",
        )

    def test_ai(self):
        data = self.data

        c_ai_nl_value = (
            data[
                (data["Croho groepeernaam"] == "B Artificial Intelligence")
                & (data["Herkomst"] == "NL")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_cumulative"].values[0]
        c_ai_eer_value = (
            data[
                (data["Croho groepeernaam"] == "B Artificial Intelligence")
                & (data["Herkomst"] == "EER")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_cumulative"].values[0]
        i_ai_nl_value = (
            data[
                (data["Croho groepeernaam"] == "B Artificial Intelligence")
                & (data["Herkomst"] == "NL")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_individual"].values[0]
        i_ai_eer_value = (
            data[
                (data["Croho groepeernaam"] == "B Artificial Intelligence")
                & (data["Herkomst"] == "EER")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_individual"].values[0]

        # self.assertEqual(
        #     c_ai_nl_value,
        #     47,
        #     "NL B Artificial Intelligence value (cumulative) is predicted wrongly ("
        #     + str(c_ai_nl_value)
        #     + " vs. 47)",
        # )
        self.assertEqual(
            c_ai_eer_value,
            21,
            "NL B Artificial Intelligence value (cumulative) is predicted wrongly ("
            + str(c_ai_eer_value)
            + " vs. 21)",
        )
        self.assertTrue(
            93 <= i_ai_nl_value <= 103,
            "NL B Artificial Intelligence value (individual) is predicted wrongly ("
            + str(i_ai_nl_value)
            + " vs. [93,103])",
        )
        self.assertTrue(
            20 <= i_ai_eer_value <= 30,
            "NL B Artificial Intelligence value (individual) is predicted wrongly ("
            + str(i_ai_eer_value)
            + " vs. [20,30])",
        )

    def test_psychologie(self):
        data = self.data

        c_psychologie_nl_value = (
            data[
                (data["Croho groepeernaam"] == "M Psychologie")
                & (data["Herkomst"] == "NL")
                & (data["Examentype"] == "Master")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_cumulative"].values[0]
        c_psychologie_eer_value = (
            data[
                (data["Croho groepeernaam"] == "M Psychologie")
                & (data["Herkomst"] == "EER")
                & (data["Examentype"] == "Master")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_cumulative"].values[0]
        i_psychologie_nl_value = (
            data[
                (data["Croho groepeernaam"] == "M Psychologie")
                & (data["Herkomst"] == "NL")
                & (data["Examentype"] == "Master")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_individual"].values[0]
        i_psychologie_eer_value = (
            data[
                (data["Croho groepeernaam"] == "M Psychologie")
                & (data["Herkomst"] == "EER")
                & (data["Examentype"] == "Master")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_individual"].values[0]

        self.assertEqual(
            c_psychologie_nl_value,
            125,
            "NL M Psychologie value (cumulative) is predicted wrongly ("
            + str(c_psychologie_nl_value)
            + " vs. 125)",
        )
        self.assertEqual(
            c_psychologie_eer_value,
            33,
            "EER M Psychologie value (cumulative) is predicted wrongly ("
            + str(c_psychologie_eer_value)
            + " vs. 33)",
        )
        self.assertTrue(
            145 <= i_psychologie_nl_value <= 165,
            "NL M Psychologie value (individual) is predicted wrongly ("
            + str(i_psychologie_nl_value)
            + " vs. [145,165])",
        )
        self.assertTrue(
            25 <= i_psychologie_eer_value <= 35,
            "EER M Psychologie value (individual) is predicted wrongly ("
            + str(i_psychologie_eer_value)
            + " vs. [25,35])",
        )
