import pandas as pd
import data_reading
import numpy as np
import plotnine as pn
from scipy.stats import ttest_ind, chi2_contingency


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

# Unusued method, initially used for describing data
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
    """
    Takes in a drug and a trait. Returns dataframe, where drug usage
    numbers have been counted per each drug usage classifier for the given trait
    """
    output_data = data[[trait, drug]]
    output_data = output_data.groupby([trait, drug]).size().reset_index(name="Amount")

    return output_data


def get_all_drug_usage_by_trait(data: pd.DataFrame, trait: str):
    """
    Same as get_drug_usage_by_trait, but gets the data about all the drugs.
    Melts them into a DataFrame with columns "[Trait Name], Drug, Classifier and Users"
    """
    output_data = data[data_reading.DRUG_TYPES + [trait]]

    grouped_usage = pd.DataFrame(columns=[trait, "Drug", "Classifier", "Users"])

    # Manual melt due to the abundance of columns
    for drug_type in data_reading.DRUG_TYPES:
        single_drug_grouping = output_data[[trait, drug_type]]
        single_drug_grouping = single_drug_grouping.groupby([trait, drug_type], observed=True).size().reset_index(name="Users")
        single_drug_grouping = single_drug_grouping.rename(columns={drug_type: "Classifier"})
        single_drug_grouping["Drug"] = drug_type

        grouped_usage = pd.concat(
            (grouped_usage, single_drug_grouping)
        )

    return grouped_usage


def get_uk_and_us_cannabis_users(data: pd.DataFrame):
    """
    Gets proportional number of cannabis users for USA and UK
    Calculated with respect to the country
    """ 
    cannabis_users_by_country = data.groupby(["Country", "Cannabis"]).size().reset_index(name="Amount")
    # Get all users
    users_amount_by_country = (
        cannabis_users_by_country[cannabis_users_by_country["Cannabis"] != "Never Used"].drop("Cannabis", axis=1)
            .groupby(["Country"]).sum().reset_index()
    )
    users_amount_by_country["Type"] = "User"

    # Add all non-users
    users_amount_by_country = pd.concat((
        users_amount_by_country,
        cannabis_users_by_country[cannabis_users_by_country["Cannabis"] == "Never Used"]
            .rename(columns={"Cannabis": "Type"}),
    ))

    # Get total amount of datapoints per each country
    total_usa_datapoints = users_amount_by_country.drop("Type", axis=1).groupby(["Country"]).sum().loc["USA"]["Amount"]
    total_uk_datapoints = users_amount_by_country.drop("Type", axis=1).groupby(["Country"]).sum().loc["UK"]["Amount"]

    # Initialize a new column
    users_amount_by_country["Proportion"] = 0.00

    # Get proportional values for both
    users_amount_by_country.loc[users_amount_by_country["Country"] == "USA", "Proportion"] = (
        users_amount_by_country.loc[users_amount_by_country["Country"] == "USA", "Amount"] / total_usa_datapoints
    )
    users_amount_by_country.loc[users_amount_by_country["Country"] == "UK", "Proportion"] = (
        users_amount_by_country.loc[users_amount_by_country["Country"] == "UK", "Amount"] / total_uk_datapoints
    )

    return users_amount_by_country


def get_personality_drug_melt(data: pd.DataFrame):
    """
    Takes in a pre-melted DataFrame, melts drug columns under "Drug" and "Classifier" columns
    for each drug.
    """
    grouped_usage = pd.DataFrame(columns=["Drug", "Trait", "Classifier", "Score"])
    for drug_type in data_reading.DRUG_TYPES:
        single_drug_grouping = data[["Trait", drug_type, "Score"]]
        single_drug_grouping = single_drug_grouping.rename(columns={drug_type: "Classifier"})
        single_drug_grouping["Drug"] = drug_type
        #print(single_drug_grouping)

        grouped_usage = pd.concat(
            (grouped_usage, single_drug_grouping)
        )

    return grouped_usage


def calculate_mean_personality_scores_per_drug(melted_data: pd.DataFrame):
    """
    Calculates personality indicator means for each drug.
    Also calculates CI values for each of them
    """
    mean_scores = melted_data
    mean_scores["Classifier"] = "" # Will later be overwritten to compare to overall mean

    agg = (
            mean_scores.drop(columns=["Classifier"]).groupby(["Drug", "Trait"], as_index=False)
            .agg(
                Mean=("Score", "mean"),
                SD=("Score", "std"),
                N=("Score", "size"),
            )
        )

    # standard error
    agg["SE"] = agg["SD"] / np.sqrt(agg["N"])

    # 95% CI using normal approximation (z = 1.96)
    z = 1.96
    agg["CI_low"] = agg["Mean"] - z * agg["SE"]
    agg["CI_high"] = agg["Mean"] + z * agg["SE"]

    mean_scores = mean_scores.groupby(["Drug", "Trait", "Classifier"]).mean().reset_index()

    mean_scores = mean_scores.merge(
        agg,
        on=["Drug", "Trait"]
    )

    # Rename stuff for ease of understanding
    mean_scores = mean_scores.rename(columns={"Classifier": "Type"})

    return mean_scores


def calculate_overall_personality_means(melted_data: pd.DataFrame, mean_scores: pd.DataFrame):
    """
    Calculates mean personality from all of the valid respondents, including non-users.
    Also creates respective entries for each Trait for overall mean values, alongside CI scores
    """
    melted_drugs = melted_data
    for trait in data_reading.PERSONALITY_TRAITS:
        trait_overall_mean = melted_drugs[melted_drugs["Trait"] == trait]["Score"].mean()
        overall_agg = melted_drugs[melted_drugs["Trait"] == trait]["Score"].agg(
                Mean="mean",
                SD="std",
                N="size",
            )
        
        # standard error
        overall_agg["SE"] = overall_agg["SD"] / np.sqrt(overall_agg["N"])

        # 95% CI using normal approximation (z = 1.96)
        z = 1.96
        overall_agg["CI_low"] = overall_agg["Mean"] - z * overall_agg["SE"]
        overall_agg["CI_high"] = overall_agg["Mean"] + z * overall_agg["SE"]
        mean_scores.loc[-1] = [
            "Overall mean", trait, "Overall mean", trait_overall_mean, 
            overall_agg["Mean"],
            overall_agg["SD"],
            overall_agg["N"],
            overall_agg["SE"],
            overall_agg["CI_low"],
            overall_agg["CI_high"],
        ]
        mean_scores.index = mean_scores.index + 1
        mean_scores = mean_scores.sort_index()
    
    return mean_scores


def calculate_significant_difference_from_overall(mean_scores: pd.DataFrame):
    """
    Compares overall mean's low and high CI against each other mean's low and high CI's
    Adds 2 columns to indicate mean's difference significance from overall
    replaces type's with the mean's significance
    """
    for trait in data_reading.PERSONALITY_TRAITS:
        mask = mean_scores["Trait"] == trait

        # Get Overall mean values into variables
        overall = mean_scores[(mean_scores.loc[mask, "Drug"] == "Overall mean") & mask]
        overall_mean = overall["Mean"].iloc[0]
        overall_ci_high = overall["CI_high"].iloc[0]
        overall_ci_low = overall["CI_low"].iloc[0]

        # Compare if there is significant difference
        mean_scores.loc[mask, "MeanSignificantlyHigherThanOverallMean"] = mean_scores.loc[mask, "CI_low"] > overall_ci_high
        mean_scores.loc[mask, "MeanSignificantlyLowerThanOverallMean"] = mean_scores.loc[mask, "CI_high"] < overall_ci_low

        # Calculate mean difference
        mean_scores.loc[mask, "MeanDifferenceFromOverallMean"] = (
            (mean_scores.loc[mask, "Mean"] - overall_mean).abs()
        )

        # Assign types
        mean_scores.loc[mask, "Type"] = np.where(
            mean_scores.loc[mask, "Drug"] == "Overall mean",
            "Overall mean", 
            np.where(
                mean_scores.loc[mask, "MeanSignificantlyLowerThanOverallMean"] | mean_scores.loc[mask, "MeanSignificantlyHigherThanOverallMean"],
                "Significant Difference",
                "Within Bounds"
            )
        )
    
    return mean_scores
#endregion


#region Hannese asjad

def compute_edu_level_country_ci(
    df: pd.DataFrame,
    countries=("UK", "USA"),
    edu_labels=None,
):
    """
    For each education level and country, compute:
      - proportion of respondents with that level (Mean)
      - SD, SE, normal-approx 95% CI
      - raw counts (LevelCount, TotalCountryN)

    Returns a DataFrame with columns:
      Education, Country, Mean, SD, N, SE, CI_low, CI_high,
      LevelCount, TotalCountryN
    """
    df = df[df["Country"].isin(countries)].copy()

    # Use EDUCATION order if not specified
    if edu_labels is None:
        edu_labels = list(data_reading.EDUCATION.values())

    all_rows = []
    for country in countries:
        sub = df[df["Country"] == country]
        total_n = len(sub)
        if total_n == 0:
            continue

        for edu in edu_labels:
            level_count = (sub["Education"] == edu).sum()
            p = level_count / total_n

            # Binomial approx
            sd = np.sqrt(p * (1 - p)) if total_n > 0 else np.nan
            n = total_n  # denominator for the proportion
            se = sd / np.sqrt(n) if n > 0 else np.nan
            z = 1.96
            ci_low = p - z * se
            ci_high = p + z * se

            all_rows.append(
                {
                    "Education": edu,
                    "Country": country,
                    "Mean": p,
                    "SD": sd,
                    "N": n,
                    "SE": se,
                    "CI_low": ci_low,
                    "CI_high": ci_high,
                    "LevelCount": level_count,
                    "TotalCountryN": total_n,
                }
            )

    stats = pd.DataFrame(all_rows)

    # keep a stable order on x-axis
    stats["Education"] = pd.Categorical(
        stats["Education"],
        categories=edu_labels,
        ordered=True,
    )

    return stats


def test_edu_level_country_diff(
    df: pd.DataFrame,
    level: str,
    countries=("UK", "USA"),
):
    """
    For a given education level, test if the proportion differs between countries.

    Constructs a 2x2 table:
      Country x {HasThisLevel, DoesNotHaveThisLevel}
    and runs chi-square test.

    Returns (chi2, p_value, significant_bool).
    """
    df = df[df["Country"].isin(countries)].copy()

    table = []
    for country in countries:
        sub = df[df["Country"] == country]
        has_level = (sub["Education"] == level).sum()
        not_level = len(sub) - has_level
        table.append([has_level, not_level])

    # Convert list → DataFrame
    df = pd.DataFrame(table)

    # Remove rows where all values are 0
    df = df.loc[df.sum(axis=1) > 0]

    # Remove columns where all values are 0
    df = df.loc[:, df.sum(axis=0) > 0]

    # Convert back to numpy array for chi-square
    clean_table = df.to_numpy()
    chi2, p_val, dof, expected = chi2_contingency(clean_table)
    return chi2, p_val, bool(p_val < 0.05)


def compute_all_edu_levels_country_ci(
    df: pd.DataFrame,
    countries=("UK", "USA"),
    edu_labels=None,
):
    """
    For all education levels, compute:
      - proportion stats by country (Mean, CI, etc.)
      - chi-square p-value and significance per level.

    Returns a long DataFrame for plotting.
    """
    if edu_labels is None:
        edu_labels = list(data_reading.EDUCATION.values())

    stats = compute_edu_level_country_ci(
        df,
        countries=countries,
        edu_labels=edu_labels,
    )

    # significance per level
    test_rows = []
    for level in edu_labels:
        chi2, p_val, sig = test_edu_level_country_diff(
            df,
            level=level,
            countries=countries,
        )
        test_rows.append(
            {
                "Education": level,
                "p_value": p_val,
                "Significant_p<0.05": sig,
            }
        )

    tests = pd.DataFrame(test_rows)

    stats = stats.merge(tests, on="Education", how="left")
    return stats


def compute_trait_country_ci(df: pd.DataFrame, trait: str, countries=("UK", "USA")):
    """
    Compute mean, SD, SE and 95% CI for a given trait by Country.
    Returns a DataFrame with:
      Trait, Country, Mean, SD, N, SE, CI_low, CI_high
    """
    df = df[df["Country"].isin(countries)].copy()

    rows = []
    for country in countries:
        s = df.loc[df["Country"] == country, trait].dropna()
        if s.empty:
            continue

        mean = s.mean()
        sd = s.std()
        n = s.size
        se = sd / np.sqrt(n)
        z = 1.96
        ci_low = mean - z * se
        ci_high = mean + z * se

        rows.append(
            {
                "Trait": trait,
                "Country": country,
                "Mean": mean,
                "SD": sd,
                "N": n,
                "SE": se,
                "CI_low": ci_low,
                "CI_high": ci_high,
            }
        )

    return pd.DataFrame(rows)


def test_trait_country_diff(df: pd.DataFrame, trait: str, countries=("UK", "USA")):
    """
    Welch t-test for given trait: Country[0] vs Country[1].
    Returns (t_stat, p_value, significant_bool).
    """
    g1 = df.loc[df["Country"] == countries[0], trait].dropna()
    g2 = df.loc[df["Country"] == countries[1], trait].dropna()

    if g1.empty or g2.empty:
        return np.nan, np.nan, False

    t_stat, p_val = ttest_ind(g1, g2, equal_var=False)
    return t_stat, p_val, bool(p_val < 0.05)


def compute_all_traits_country_ci(
    df: pd.DataFrame,
    traits=None,
    countries=("UK", "USA"),
):
    """
    For all traits, compute mean, SD, SE, 95% CI by Country,
    plus Welch t-test p-value and significance per trait.
    Returns one long DataFrame for a single combined plot.
    """
    if traits is None:
        traits = data_reading.PERSONALITY_TRAITS

    all_stats = []
    all_tests = []

    for trait in traits:
        stats_trait = compute_trait_country_ci(df, trait, countries=countries)
        all_stats.append(stats_trait)

        _, p_val, sig = test_trait_country_diff(df, trait, countries=countries)
        all_tests.append(
            {
                "Trait": trait,
                "p_value": p_val,
                "Significant_p<0.05": sig,
            }
        )

    stats_all = pd.concat(all_stats, ignore_index=True)
    tests_all = pd.DataFrame(all_tests)

    stats_all = stats_all.merge(tests_all, on="Trait", how="left")

    # Stable order on x-axis
    stats_all["Trait"] = pd.Categorical(
        stats_all["Trait"], categories=list(traits), ordered=True
    )

    return stats_all


def compute_drug_usage_country_ci(df: pd.DataFrame, drug: str, countries=("UK", "USA")):
    """
    For a given drug, compute proportion 'ever used' (vs never) by Country,
    plus SD, SE and normal-approx 95% CI.
    """
    df = df[df["Country"].isin(countries)].copy()
    df["EverUsed"] = (df[drug] != "Never Used").astype(int)

    rows = []
    for country in countries:
        s = df.loc[df["Country"] == country, "EverUsed"].dropna()
        if s.empty:
            continue

        mean = s.mean()  # proportion ever used
        sd = s.std()
        n = s.size
        se = sd / np.sqrt(n)
        z = 1.96
        ci_low = mean - z * se
        ci_high = mean + z * se

        rows.append(
            {
                "Drug": drug,
                "Country": country,
                "Mean": mean,
                "SD": sd,
                "N": n,
                "SE": se,
                "CI_low": ci_low,
                "CI_high": ci_high,
            }
        )

    return pd.DataFrame(rows)


def test_drug_usage_country_diff(df: pd.DataFrame, drug: str, countries=("UK", "USA")):
    """
    Chi-square test of independence for EverUsed vs Country.
    Returns (chi2, p_value, significant_bool).
    """
    df = df[df["Country"].isin(countries)].copy()
    df["EverUsed"] = (df[drug] != "Never Used").astype(int)

    table = []
    for country in countries:
        sub = df[df["Country"] == country]
        used = int((sub["EverUsed"] == 1).sum())
        never = int((sub["EverUsed"] == 0).sum())
        table.append([used, never])

    chi2, p, dof, expected = chi2_contingency(table)
    return chi2, p, bool(p < 0.05)


def plot_drug_usage_country(
    df: pd.DataFrame,
    drug: str,
    countries=("UK", "USA"),
    ref_country="UK",
):
    """
    Barplot of 'ever used' proportion for a drug by Country,
    with 95% CI, ±2 SD band of ref_country, and chi-square significance star.
    """

    stats = compute_drug_usage_country_ci(df, drug, countries=countries)
    if stats.empty:
        raise ValueError("No data for given drug/countries.")

    # Reference ±2 SD band (based on 0/1 EverUsed coding)
    ref_row = stats[stats["Country"] == ref_country]
    if ref_row.empty:
        raise ValueError(f"Reference country {ref_country} not found in data.")
    ref_row = ref_row.iloc[0]
    band_low = ref_row["Mean"] - 2 * ref_row["SD"]
    band_high = ref_row["Mean"] + 2 * ref_row["SD"]

    _, p_val, significant = test_drug_usage_country_diff(df, drug, countries=countries)

    y_star = stats["CI_high"].max() + 0.05
    sig_df = pd.DataFrame(
        {
            "x": [1.5],
            "y": [y_star],
            "label": ["*" if significant else ""],
        }
    )

    p = (
        pn.ggplot(stats, pn.aes("Country", "Mean"))
        + pn.geom_col(fill="#0C475B")
        + pn.geom_errorbar(pn.aes(ymin="CI_low", ymax="CI_high"), width=0.2)
        + pn.geom_hline(yintercept=band_low, linetype="dashed", color="#D55E00")
        + pn.geom_hline(yintercept=band_high, linetype="dashed", color="#D55E00")
        + pn.geom_text(
            pn.aes(x="x", y="y", label="label"),
            data=sig_df,
            inherit_aes=False,
            size=18,
        )
        + pn.scale_y_continuous(labels=lambda v: [f"{vv:.0%}" for vv in v])
        + pn.labs(
            title=(
                f"{drug}: ever used in UK vs USA "
                f"(95% CI, ±2 SD of {ref_country}; p = {p_val:.3f})"
            ),
            x="Country",
            y="Proportion ever used",
        )
        + pn.theme(
            figure_size=(6, 4),
            axis_text_x=pn.element_text(rotation=0),
        )
    )

    return p


def compute_all_drug_usage_country_ci(
    df: pd.DataFrame,
    drugs=None,
    countries=("UK", "USA"),
):
    """
    For all drugs, compute 'ever used' proportion stats by Country,
    plus chi-square p-value and significance flag per drug.
    Returns one long DataFrame suitable for a single combined plot.
    """
    if drugs is None:
        drugs = data_reading.DRUG_TYPES

    all_stats = []
    all_tests = []

    for drug in drugs:
        # reuse your per-drug summary function
        stats = compute_drug_usage_country_ci(df, drug, countries=countries)
        all_stats.append(stats)

        # reuse your chi-square significance test
        chi2, p_val, sig = test_drug_usage_country_diff(df, drug, countries=countries)
        all_tests.append(
            {
                "Drug": drug,
                "p_value": p_val,
                "Significant_p<0.05": sig,
            }
        )

    stats_all = pd.concat(all_stats, ignore_index=True)
    tests_all = pd.DataFrame(all_tests)

    # merge p-values and significance flags into stats
    stats_all = stats_all.merge(tests_all, on="Drug", how="left")

    # keep a stable drug order
    if drugs is not None:
        stats_all["Drug"] = pd.Categorical(
            stats_all["Drug"], categories=list(drugs), ordered=True
        )

    return stats_all


def plot_all_drug_usage_country(
    df: pd.DataFrame,
    drugs=None,
    countries=("UK", "USA"),
):
    """
    Single plot:
      x-axis: Drug
      bars: proportion 'ever used' (Mean) for UK vs USA
      error bars: 95% CI
      * above a drug name if UK vs USA differ significantly (p < 0.05 chi-square).
    """
    if drugs is None:
        drugs = data_reading.DRUG_TYPES

    stats = compute_all_drug_usage_country_ci(df, drugs=drugs, countries=countries)

    # star positions: per drug, above the highest CI_high across countries
    star_rows = (
        stats.groupby("Drug")["CI_high"]
        .max()
        .reset_index()
        .rename(columns={"CI_high": "y"})
    )

    sig_flags = (
        stats[["Drug", "Significant_p<0.05"]]
        .drop_duplicates("Drug")
    )
    star_rows = star_rows.merge(sig_flags, on="Drug", how="left")
    star_rows["label"] = star_rows["Significant_p<0.05"].map(lambda x: "*" if x else "")

    p = (
        pn.ggplot(stats, pn.aes("Drug", "Mean", fill="Country"))
        + pn.geom_col(
            position=pn.position_dodge(width=0.8),
            width=0.7,
        )
        + pn.geom_errorbar(
            pn.aes(ymin="CI_low", ymax="CI_high"),
            position=pn.position_dodge(width=0.8),
            width=0.25,
        )
        # significance stars (centered over each drug)
        + pn.geom_text(
            pn.aes(x="Drug", y="y", label="label"),
            data=star_rows,
            inherit_aes=False,
            size=10,
            va="bottom",
        )
        + pn.scale_y_continuous(labels=lambda v: [f"{vv:.0%}" for vv in v])
        + pn.labs(
            title="Proportion ever used for each drug in UK vs USA\n"
                  "(95% CI, * = p < 0.05, chi-square test)",
            x="Drug",
            y="Proportion ever used",
            fill="Country",
        )
        + pn.theme(
            figure_size=(12, 6),
            axis_text_x=pn.element_text(rotation=45, hjust=1),
        )
    )

    return p


def plot_all_traits_country(
    df: pd.DataFrame,
    traits=None,
    countries=("UK", "USA"),
):
    """
    Single plot:
      x-axis: Trait
      bars: mean score for UK vs USA
      error bars: 95% CI
      * above trait name if UK vs USA differ significantly (p < 0.05).
    """
    stats = compute_all_traits_country_ci(df, traits=traits, countries=countries)

    # Where to put the stars: per trait, above highest CI_high
    star_rows = (
        stats.groupby("Trait")["CI_high"]
        .max()
        .reset_index()
        .rename(columns={"CI_high": "y"})
    )

    sig_flags = (
        stats[["Trait", "Significant_p<0.05"]]
        .drop_duplicates("Trait")
    )
    star_rows = star_rows.merge(sig_flags, on="Trait", how="left")
    star_rows["label"] = star_rows["Significant_p<0.05"].map(lambda x: "*" if x else "")
    # little vertical margin above the bars
    star_rows["y"] = star_rows["y"] + 0.4

    p = (
        pn.ggplot(stats, pn.aes("Trait", "Mean", fill="Country"))
        + pn.geom_col(
            position=pn.position_dodge(width=0.8),
            width=0.7,
        )
        + pn.geom_errorbar(
            pn.aes(ymin="CI_low", ymax="CI_high"),
            position=pn.position_dodge(width=0.8),
            width=0.25,
        )
        + pn.geom_text(
            pn.aes(x="Trait", y="y", label="label"),
            data=star_rows,
            inherit_aes=False,
            size=10,
            va="bottom",
        )
        + pn.labs(
            title="Personality traits: UK vs USA (mean ± 95% CI; * = p < 0.05)",
            x="Trait",
            y="Mean score",
            fill="Country",
        )
        + pn.theme(
            figure_size=(10, 5),
            axis_text_x=pn.element_text(rotation=45, hjust=1),
        )
    )

    return p


def plot_edu_levels_country(
    df: pd.DataFrame,
    countries=("UK", "USA"),
    edu_labels=None,
):
    """
    Single plot:
      x-axis: Education level
      bars: proportion of respondents in that level (UK vs USA)
      error bars: 95% CI
      * above level if proportions differ significantly (p < 0.05).
    """
    if edu_labels is None:
        edu_labels = list(data_reading.EDUCATION.values())

    stats = compute_all_edu_levels_country_ci(
        df,
        countries=countries,
        edu_labels=edu_labels,
    )

    # Make sure Education is a categorical with the desired left→right order
    stats["Education"] = pd.Categorical(
        stats["Education"],
        categories=edu_labels,
        ordered=True,
    )


    # positions for stars: per level, above highest CI_high
    star_rows = (
        stats.groupby("Education")["CI_high"]
        .max()
        .reset_index()
        .rename(columns={"CI_high": "y"})
    )

    sig_flags = (
        stats[["Education", "Significant_p<0.05"]]
        .drop_duplicates("Education")
    )
    star_rows = star_rows.merge(sig_flags, on="Education", how="left")
    star_rows["label"] = star_rows["Significant_p<0.05"].map(
        lambda x: "*" if x else ""
    )
    star_rows["y"] = star_rows["y"] + 0.02  # little margin above bars

    p = (
        pn.ggplot(stats, pn.aes("Education", "Mean", fill="Country"))
        + pn.geom_col(
            position=pn.position_dodge(width=0.8),
            width=0.7,
        )
        + pn.geom_errorbar(
            pn.aes(ymin="CI_low", ymax="CI_high"),
            position=pn.position_dodge(width=0.8),
            width=0.25,
        )
        + pn.geom_text(
            pn.aes(x="Education", y="y", label="label"),
            data=star_rows,
            inherit_aes=False,
            size=10,
            va="bottom",
        )
        + pn.scale_y_continuous(labels=lambda v: [f"{vv:.0%}" for vv in v])
        + pn.labs(
            title="Education distribution: UK vs USA "
                  "(proportion by level, 95% CI, * = p < 0.05)",
            x="Education level",
            y="Proportion of respondents",
            fill="Country",
        )
        + pn.theme(
            figure_size=(12, 6),
            axis_text_x=pn.element_text(rotation=45, hjust=1),
        )
    )

    return p
#endregion

#region HELPER METHODS
def _count_without_na(series: pd.Series):
    return series.dropna().count()


def _count_na(series: pd.Series):
    return series.isna().sum()
#endregion
