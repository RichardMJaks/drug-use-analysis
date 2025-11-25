# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Exploring the Link Between Personality Traits and Substance Use

# %% [markdown]
# Importing both necessary and custom modules

# %%
import sys
import pandas as pd
import plotnine as pn
import sklearn as sk
import numpy as np
sys.path.insert(1, "scripts/custom_modules") # Required to import scripts from another directory
import data_analysis
import data_reading

# %% [markdown]
# Read in the data

# %%
data = data_reading.read_data("data/drug_consumption.csv")
human_readable_data = data_reading.convert_data(data)
human_readable_data = data_reading.convert_drug_usage_classifiers(data)
human_readable_data

# %%
