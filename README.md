# Student forecasting model

This Python script predicts the influx of students at the Radboud University for the current year and week. The year and week can also be specified.

# Usage of program

Execute the script with the current year and week using the following command:

```
python main.py
```

To predict different years/weeks or in a different way, use any of the following command line parameters to your liking.

## Years and weeks specification

Execute the script with a specified year and week with `-y` and `-w`, e.g.:

```
python main.py -w 6 -y 2024
python main.py -W 1 2 3 -Y 2024
python main.py -year 2023 2024
python main.py -week 40 41
```

For predicting multiple years/weeks we also provide slicing, in the example the weeks 10 up to and including 20 will be predicted:

```
python main.py -w 10 : 20 -y 2023
```

## Datasets

The main datasets that are used in this script are the cumulative data per programme/origin/year/week and individual data per student. If one of these is not present then only the other dataset can be used.

```
python main.py -d individual
python main.py -D cumulative
python main.py -dataset both
```

## Configuration

The script has to have a configuration file for numerous different reason. The path to the configuration is `configuration/configuration.json` by default, but can be changed in the terminal:

```
python main.py -c path/to/configuration.json
python main.py -configuration longer/path/to/config.json
```

## Student year prediction

By default only the first year students are predicted, while the script also supports prediction of higher years or volume predictions.

```
python main.py -sy first-years
python main.py -SY higher-years
python main.py -studentyear volume
```

## Postprocess subset

By default when a file with the latest information is added, the postprocessing (consisting of calculating the ensembles and error metrics) is done on all the data. For time purposes, there exists an option to only postprocess the predicted data.

```
python main.py -p all
python main.py -postprocess new
```

## Syntax

In the following the syntax for all command line options is shown:

| Cmd setting             | Short notation | Large notation | Cmd option             | Short notation | Large notation |
|-------------------------|----------------|----------------|------------------------|----------------|----------------|
| Prediction years        | -y or -Y       | -year          | One or more years      | 1 2 3          | 1 : 3          |
| Prediction weeks        | -w or -W       | -week          | One or more weeks      | 10 11 12       | 10 : 12        |
| Dataset                 | -d or -D       | -dataset       | Only individual        | i              | individual     |
| Dataset                 | -d or -D       | -dataset       | Only cumulative        | c              | cumulative     |
| Dataset                 | -d or -D       | -dataset       | Both dataset           | b              | both           |
| Configuration           | -c or -C       | -configuration | Config file path (str) |                |                |
| Student year prediction | -sy or -SY     | -studentyear   | First years            | f              | first-years    |
| Student year prediction | -sy or -SY     | -studentyear   | Higher years           | h              | higher-years   |
| Student year prediction | -sy or -SY     | -studentyear   | Volume                 | v              | volume         |
| Postprocess subset      | -p or -P       | -postprocess   | All                    | a              | all            |
| Postprocess subset      | -p or -P       | -postprocess   | New                    | n              | new            |

## Large example

Example 1: Predict volume of year 2023 and 2024, weeks 10 up to and including 20, use both datasets, only postprocess new predicted data

```
python main.py -y 2023 2024 -w 10 : 20 -d b -sy v -p n
```

# Process of script

The following image depicts the process of the script when executed with available datasets.

![Process of script](doc/ActivityDiagrams/NewProcess/new_process_script_with_legend.png)