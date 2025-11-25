import pandas as pd

#region CORE METHODS
def get_columns_unique_value_counts(data: pd.DataFrame):
    """
    Counts the number of each unique value in DataFrame
    """

    data = data.apply(pd.Series.value_counts)
    
    return data
    

def get_columns_value_counts(data: pd.DataFrame):
    """
    Counts the number of values in DataFrame
    """

    data = data.apply(_count_without_na)

    return data


def get_columns_na_counts(data: pd.DataFrame):
    """
    Count the number of NaN values in DataFrame
    """
    data.apply(_count_na)


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
#endregion


#region HELPER METHODS
def _count_without_na(series: pd.Series):
    return series.dropna().count()


def _count_na(series: pd.Series):
    return series.isna().sum()
#endregion
