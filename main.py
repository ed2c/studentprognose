import datetime
import sys
from enum import Enum

from scripts.load_data import *
from scripts.availabledata import *
from scripts.individual import *
from scripts.cumulative import *
from scripts.bothdatasets import *
from scripts.dataoption import *

class Main:
    def __init__(self, arguments):
        self._parse_arguments(arguments)

    class Cmd(Enum):
        NO_ARG = 0
        WEEKS = 1
        YEARS = 2
        DATASETS = 3
        CONFIGURATION = 4
        STUDENT_YEAR_PREDICTION = 5
        POST_PROCESS = 6

    def _parse_arguments(self, arguments):
        cmd_arg = Main.Cmd.NO_ARG

        self.weeks = []
        self.week_slice = False
        self.years = []
        self.year_slice = False
        self.data_option = DataOption.BOTH_DATASETS
        self.configuration_path = "configuration/configuration.json"
        self.student_year_prediction = StudentYearPrediction.FIRST_YEARS
        self.postprocess_subset = PostProcessSubset.ALL

        try:
            # First arguments is always the name of the python script
            for i in range(1, len(arguments)):
                arg = arguments[i]

                if cmd_arg == Main.Cmd.WEEKS and arg == ":":
                    self.week_slice = True
                elif cmd_arg == Main.Cmd.WEEKS and arg.isnumeric():
                    if self.week_slice:
                        last_week = self.weeks.pop(-1)
                        self.weeks = self.weeks + list(range(last_week, int(arg) + 1))
                        self.week_slice = False
                    else:
                        self.weeks.append(int(arg))
                elif cmd_arg == Main.Cmd.YEARS and arg == ":":
                    self.year_slice = True
                elif cmd_arg == Main.Cmd.YEARS and arg.isnumeric():
                    if self.year_slice:
                        last_year = self.years.pop(-1)
                        self.years = self.years + list(range(last_year, int(arg) + 1))
                        self.year_slice = False
                    else:
                        self.years.append(int(arg))
                elif cmd_arg == Main.Cmd.DATASETS:
                    if arg == "i" or arg == "individual":
                        self.data_option = DataOption.INDIVIDUAL
                        cmd_arg = Main.Cmd.NO_ARG
                    elif arg == "c" or arg == "cumulative":
                        self.data_option = DataOption.CUMULATIVE
                        cmd_arg = Main.Cmd.NO_ARG
                    elif arg == "b" or arg == "both":
                        self.data_option = DataOption.BOTH_DATASETS
                        cmd_arg = Main.Cmd.NO_ARG
                elif cmd_arg == Main.Cmd.CONFIGURATION:
                    if os.path.exists(arg):
                        self.configuration_path = arg
                    else:
                        raise Exception("Configuration path does not exist")
                    cmd_arg = Main.Cmd.NO_ARG
                elif cmd_arg == Main.Cmd.STUDENT_YEAR_PREDICTION:
                    if arg == "f" or arg == "first-years":
                        self.student_year_prediction = StudentYearPrediction.FIRST_YEARS
                        cmd_arg = Main.Cmd.NO_ARG
                    elif arg == "h" or arg == "higher-years":
                        self.student_year_prediction = StudentYearPrediction.HIGHER_YEARS
                        cmd_arg = Main.Cmd.NO_ARG
                    elif arg == "v" or arg == "volume":
                        self.student_year_prediction = StudentYearPrediction.VOLUME
                        cmd_arg = Main.Cmd.NO_ARG
                elif cmd_arg == Main.Cmd.POST_PROCESS:
                    if arg == "a" or arg == "all":
                        self.postprocess_subset = PostProcessSubset.ALL
                        cmd_arg = Main.Cmd.NO_ARG
                    if arg == "n" or arg == "new":
                        self.postprocess_subset = PostProcessSubset.NEW
                        cmd_arg = Main.Cmd.NO_ARG


                if arg == "-w" or arg == "-W" or arg == "-week":
                    cmd_arg = Main.Cmd.WEEKS
                elif arg == "-y" or arg == "-Y" or arg == "-year":
                    cmd_arg = Main.Cmd.YEARS
                elif arg == "-d" or arg == "-D" or arg == "-dataset":
                    cmd_arg = Main.Cmd.DATASETS
                elif arg == "-c" or arg == "-C" or arg == "-configuration":
                    cmd_arg = Main.Cmd.CONFIGURATION
                elif arg == "-sy" or arg == "-SY" or arg == "-studentyear":
                    cmd_arg = Main.Cmd.STUDENT_YEAR_PREDICTION
                elif arg == "-p" or arg == "-P" or arg == "-postprocess":
                    cmd_arg = Main.Cmd.POST_PROCESS
        except:
            print("Something went wrong while parsing the arguments, read the README.md for usage.")


        weeks_specified = True
        if self.weeks == []:
            weeks_specified = False
            current_week = datetime.date.today().isocalendar()[1]
            # Max of 52 weeks, week 53 is an edge case where the user should manually input data
            if current_week > 52:
                print("Current week is week 53, check what weeknumber should be used")
                print("Now predicting for week 52")
                current_week = 52
            self.weeks = [current_week]

        if self.years == []:
            current_year = datetime.date.today().year
            
            if not weeks_specified and self.weeks[0] >= 40:
                current_year += 1

            self.years = [current_year]

    def run(self):
        print("Predicting for years: ", self.years, " and weeks: ", self.weeks)

        # Load configuration
        print("Loading configuration...")
        self.configuration = load_configuration(self.configuration_path)

        # Load data
        print("Loading data...")
        self.data_individual, self.data_cumulative, self.data_studentcount, self.data_latest, self.data_distances, self.ensemble_weights = load_data(self.configuration, self.student_year_prediction)

        # Initialize dataholder
        self.dataholder = None
        if self.data_option == DataOption.BOTH_DATASETS:
            if self.data_individual is None:
                raise Exception("Individual dataset not found")
            if self.data_cumulative is None:
                raise Exception("Cumulative dataset not found")
            self.dataholder = BothDatasets(self.data_individual, self.data_cumulative, self.data_distances, self.data_studentcount, self.configuration, self.student_year_prediction)
        elif self.data_option == DataOption.INDIVIDUAL:
            if self.data_individual is None:
                raise Exception("Individual dataset not found")
            self.dataholder = Individual(self.data_individual, self.data_distances, self.configuration)
        elif self.data_option == DataOption.CUMULATIVE:
            if self.data_cumulative is None:
                raise Exception("Cumulative dataset not found")
            self.dataholder = Cumulative(self.data_cumulative, self.data_studentcount, self.configuration, self.student_year_prediction)

        # Preprocess data
        print("Preprocessing...")
        self.data_cumulative = self.dataholder.preprocess()

        # Initialize data total class
        CWD = os.path.dirname(os.path.abspath(__file__))
        self.dataholder.data_total.initialize(self.data_latest, self.ensemble_weights, self.data_studentcount, self.configuration, CWD)

        # Set programme and origin filtering
        self.dataholder.set_filtering(self.configuration["filtering"]["programme"], self.configuration["filtering"]["herkomst"])

        for year in self.years:
            for week in self.weeks:
                # Run SARIMA models
                if self.student_year_prediction == StudentYearPrediction.VOLUME:
                    data_to_predict, predicted_data_first_years = self.dataholder.predict_nr_of_students(year, week, StudentYearPrediction.FIRST_YEARS)
                    if data_to_predict is None:
                        continue
                    data_first_years = self.dataholder.data_total.append_predicted_data(data_to_predict, predicted_data_first_years, StudentYearPrediction.FIRST_YEARS)

                    data_to_predict, predicted_data_higher_years = self.dataholder.predict_nr_of_students(year, week, StudentYearPrediction.HIGHER_YEARS)
                    if data_to_predict is None:
                        continue
                    data_higher_years = self.dataholder.data_total.append_predicted_data(data_to_predict, predicted_data_higher_years, StudentYearPrediction.HIGHER_YEARS)

                    data = calculate_volume_predicted_data(data_first_years, data_higher_years, year, week)
                else:
                    data_to_predict, predicted_data = self.dataholder.predict_nr_of_students(year, week, self.student_year_prediction)
                    if data_to_predict is None:
                        continue
                    data = self.dataholder.data_total.append_predicted_data(data_to_predict, predicted_data, self.student_year_prediction)
                
                self.dataholder.data_total.prepare_data(data, self.data_cumulative)

                if self.data_option == DataOption.CUMULATIVE or self.data_option == DataOption.BOTH_DATASETS:
                    # Run ratio model
                    self.dataholder.data_total.predict_with_ratio(self.data_cumulative)

                # Post process
                print("Postprocessing...")
                self.dataholder.data_total.postprocess(self.postprocess_subset, year, week)

                # Save final output
                print("Saving output...")
                self.dataholder.data_total.save_output()


if __name__ == "__main__":
    main = Main(sys.argv)
    main.run()