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
#endregion


#region HELPER METHODS
def _count_without_na(series: pd.Series):
    return series.dropna().count()


def _count_na(series: pd.Series):
    return series.isna().sum()
#endregion
