# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %%
import sys

# %%
import pandas as pd
import plotnine as pn
import sklearn as sk
import numpy as np

# %%
sys.path.insert(1, "scripts/custom_modules") # Required to import scripts from another directory
import data_analysis
import data_reading

# %%
DRUG_TYPES = [
    "Alcohol",
    "Amphet",
    "Amyl",
    "Benzos",
    "Caff",
    "Cannabis",
    "Choc",
    "Coke",
    "Crack",
    "Ecstasy",
    "Heroin",
    "Ketamine",
    "Legalh",
    "LSD",
    "Meth",
    "Mushrooms",
    "Nicotine",
    "Semer",
    "VSA",
]

# %%
SEPARATOR_STRING = "\n" + "-" * 20 + "\n"


# %%
def describe_data(data: pd.DataFrame):
    print("DATA DESCRIPTION")

    print(SEPARATOR_STRING)

    print("DATA")
    print(data)

    print(SEPARATOR_STRING)

    print("DRUG CLASSIFIER COUNTS")
    drug_classifier_counts = data_analysis.get_columns_unique_value_counts(data[DRUG_TYPES])
    print(drug_classifier_counts)

    print(SEPARATOR_STRING)

    # Count not NaN values in columns
    print("TOTAL VALUES IN COLUMNS")
    row_counts = data_analysis.get_columns_value_counts(data)
    print(row_counts)

    print(SEPARATOR_STRING)

    # Count NaN values in columns
    print("NaN VALUES IN COLUMNS")
    nan_counts = data_analysis.get_columns_na_counts(data)
    print(nan_counts)
    
    print(SEPARATOR_STRING)


# %%
if __name__ == '__main__':
    data = data_reading.read_data("data/drug_consumption.csv")

    converted_data = data_reading.convert_data(data)
    converted_data = data_reading.convert_drug_usage_classifiers(data, DRUG_TYPES)
    
    drug_classifier_counts = data_analysis.get_columns_unique_value_counts(data[DRUG_TYPES])
    print(drug_classifier_counts)
    pn.ggplot(drug_classifier_counts, pn.aes(x="Alcohol", y="id")) + pn.geom_bar()

    # describe_data(data)
