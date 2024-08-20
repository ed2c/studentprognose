import unittest
import subprocess
import pandas as pd

# This test takes approximately 80 seconds.


class TestMainScript(unittest.TestCase):
    def test_run_main_script(self):
        # Define the command to be run
        command = [
            "python",
            "main.py",
            "-w",
            "12",
            "-y",
            "2024",
            "-d",
            "individual",
            "-f",
            "configuration/filtering/test.json",
        ]

        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True)

        # Check that the command was successful
        self.assertEqual(result.returncode, 0, "The script did not run successfully")

        # Optionally, check for any error output
        self.assertEqual(result.stderr, "", "There was an error in running the script")

        # Check if predicted cumulative value corresponds with correct value

        # B Bedrijfskunde
        data = pd.read_excel("data/output/output_prelim.xlsx")
        bedrijfskunde_nl_value = (
            data[
                (data["Croho groepeernaam"] == "B Bedrijfskunde")
                & (data["Herkomst"] == "NL")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_individual"].values[0]
        bedrijfskunde_eer_value = (
            data[
                (data["Croho groepeernaam"] == "B Bedrijfskunde")
                & (data["Herkomst"] == "EER")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_individual"].values[0]

        self.assertTrue(
            380 <= bedrijfskunde_nl_value <= 385,
            "NL B Bedrijfskunde value (individual) is predicted wrongly ("
            + str(bedrijfskunde_nl_value)
            + " vs. [380,385])",
        )
        self.assertTrue(
            25 <= bedrijfskunde_eer_value <= 30,
            "EER B Bedrijfskunde value (individual)  is predicted wrongly ("
            + str(bedrijfskunde_eer_value)
            + " vs. [25,30])",
        )

        # B Artificial Intelligence (to test numerus fixus)
        ai_nl_value = (
            data[
                (data["Croho groepeernaam"] == "B Artificial Intelligence")
                & (data["Herkomst"] == "NL")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_individual"].values[0]
        ai_eer_value = (
            data[
                (data["Croho groepeernaam"] == "B Artificial Intelligence")
                & (data["Herkomst"] == "EER")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_individual"].values[0]

        self.assertTrue(
            90 <= ai_nl_value <= 95,
            "NL B Artificial Intelligence value (individual) is predicted wrongly ("
            + str(ai_nl_value)
            + " vs. [90,95])",
        )
        self.assertTrue(
            17 <= ai_eer_value <= 22,
            "NL B Artificial Intelligence value (individual) is predicted wrongly ("
            + str(ai_eer_value)
            + " vs. [17,22])",
        )

        # M Psychologie (to test examentype master)
        psychologie_nl_value = (
            data[
                (data["Croho groepeernaam"] == "M Psychologie")
                & (data["Herkomst"] == "NL")
                & (data["Examentype"] == "Master")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_individual"].values[0]
        psychologie_eer_value = (
            data[
                (data["Croho groepeernaam"] == "M Psychologie")
                & (data["Herkomst"] == "EER")
                & (data["Examentype"] == "Master")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_individual"].values[0]

        self.assertTrue(
            140 <= psychologie_nl_value <= 150,
            "NL M Psychologie value (individual) is predicted wrongly ("
            + str(psychologie_nl_value)
            + " vs. [140,150])",
        )
        self.assertTrue(
            30 <= psychologie_eer_value <= 35,
            "EER M Psychologie value (individual) is predicted wrongly ("
            + str(psychologie_eer_value)
            + " vs. [30,35])",
        )


# This allows the tests to be run from the command line with `python test_script.py`
if __name__ == "__main__":
    unittest.main()
