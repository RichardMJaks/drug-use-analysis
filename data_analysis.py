import pandas as pd
import data_reading

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

def get_drug_usage_by_trait(data: pd.DataFrame, drug: str, trait: str):
    output_data = data[[trait, drug]]
    output_data = output_data.groupby([trait, drug]).size().reset_index(name="Amount")

    return output_data


def get_all_drug_usage_by_trait(data: pd.DataFrame, trait: str):
    output_data = data[data_reading.DRUG_TYPES + [trait]]

    grouped_usage = pd.DataFrame(columns=[trait, "Drug", "Classifier", "Users"])
    for drug_type in data_reading.DRUG_TYPES:
        single_drug_grouping = output_data[[trait, drug_type]]
        single_drug_grouping = single_drug_grouping.groupby([trait, drug_type], observed=True).size().reset_index(name="Users")
        single_drug_grouping = single_drug_grouping.rename(columns={drug_type: "Classifier"})
        single_drug_grouping["Drug"] = drug_type

        grouped_usage = pd.concat(
            (grouped_usage, single_drug_grouping)
        )

    return grouped_usage
	
#endregion


#region HELPER METHODS
def _count_without_na(series: pd.Series):
    return series.dropna().count()


def _count_na(series: pd.Series):
    return series.isna().sum()
#endregion
