from enum import Enum


class DataOption(Enum):
        INDIVIDUAL = 1
        CUMULATIVE = 2
        BOTH_DATASETS = 3

class StudentYearPrediction(Enum):
        FIRST_YEARS = 1
        HIGHER_YEARS = 2
        VOLUME = 3

class PostProcessSubset(Enum):
        ALL = 0
        NEW = 1