import unittest
import subprocess
import pandas as pd

# This test takes approximately 75 seconds.


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
            "configuration/filtering/fast_test.json",
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

        self.assertTrue(
            380 <= bedrijfskunde_nl_value <= 385,
            "NL B Bedrijfskunde value (individual) is predicted wrongly ("
            + str(bedrijfskunde_nl_value)
            + " vs. [380,385])",
        )


# This allows the tests to be run from the command line with `python test_script.py`
if __name__ == "__main__":
    unittest.main()
