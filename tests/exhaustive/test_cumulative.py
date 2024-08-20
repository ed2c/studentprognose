import unittest
import subprocess
import pandas as pd

# This test takes approximately 120 seconds.


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
            "cumulative",
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
        )["SARIMA_cumulative"].values[0]
        bedrijfskunde_eer_value = (
            data[
                (data["Croho groepeernaam"] == "B Bedrijfskunde")
                & (data["Herkomst"] == "EER")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_cumulative"].values[0]

        self.assertIn(
            bedrijfskunde_nl_value,
            [266, 294],
            "NL B Bedrijfskunde value (cumulative) is predicted wrongly ("
            + str(bedrijfskunde_nl_value)
            + " vs. [266 or 294])",
        )
        self.assertEqual(
            bedrijfskunde_eer_value,
            29,
            "EER B Bedrijfskunde value (cumulative) is predicted wrongly ("
            + str(bedrijfskunde_eer_value)
            + " vs. 29)",
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
        )["SARIMA_cumulative"].values[0]
        ai_eer_value = (
            data[
                (data["Croho groepeernaam"] == "B Artificial Intelligence")
                & (data["Herkomst"] == "EER")
                & (data["Examentype"] == "Bachelor")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_cumulative"].values[0]

        self.assertEqual(
            ai_nl_value,
            47,
            "NL B Artificial Intelligence value (cumulative) is predicted wrongly ("
            + str(ai_nl_value)
            + " vs. 47)",
        )
        self.assertEqual(
            ai_eer_value,
            49,
            "NL B Artificial Intelligence value (cumulative) is predicted wrongly ("
            + str(ai_eer_value)
            + " vs. 49)",
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
        )["SARIMA_cumulative"].values[0]
        psychologie_eer_value = (
            data[
                (data["Croho groepeernaam"] == "M Psychologie")
                & (data["Herkomst"] == "EER")
                & (data["Examentype"] == "Master")
                & (data["Collegejaar"] == 2024)
                & (data["Weeknummer"] == 12)
            ]
        )["SARIMA_cumulative"].values[0]

        self.assertEqual(
            psychologie_nl_value,
            48,
            "NL M Psychologie value (cumulative) is predicted wrongly ("
            + str(psychologie_nl_value)
            + " vs. 48)",
        )
        self.assertEqual(
            psychologie_eer_value,
            50,
            "EER M Psychologie value (cumulative) is predicted wrongly ("
            + str(psychologie_eer_value)
            + " vs. 50)",
        )


# This allows the tests to be run from the command line with `python test_script.py`
if __name__ == "__main__":
    unittest.main()
