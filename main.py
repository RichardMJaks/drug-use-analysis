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
# Dataset: https://www.kaggle.com/datasets/mexwell/drug-consumption-classification/data

# %% [markdown]
# Importing both necessary and custom modules

# %%
import sys
import pandas as pd
import plotnine as pn
import sklearn as sk
import numpy as np
import data_analysis
import data_reading
import matplotlib.pyplot as plt
from scipy import stats

# %% [markdown]
# Read in the data

# %%
data = data_reading.read_data("data/drug_consumption.csv")
dataOG = data.copy()
human_readable_data = data_reading.convert_data(data)
human_readable_data = data_reading.convert_drug_usage_classifiers(human_readable_data)
print(list(data_reading.EDUCATION.values()))
human_readable_data["Education"] = pd.Categorical(
        human_readable_data["Education"],
        categories=list(data_reading.EDUCATION.values()),
        ordered=True,
    )
human_readable_data

# %% [markdown]
# Make a DataFrame useful for categorizing drugs

# %%
# Bring data to a shape possible to be shown as a plot
drug_usage = data_analysis.get_columns_unique_value_counts(data[data_reading.DRUG_TYPES])
drug_usage = drug_usage.transpose().reset_index(names="Drug")
drug_usage = drug_usage.melt(id_vars="Drug", var_name="Usage", value_name="Amount")

# Replace NaN amounts with 0-s
drug_usage = drug_usage.fillna(0)

drug_usage

# %%
(
    pn.ggplot(drug_usage, pn.aes("Usage", "Amount", fill="Drug")) 
    + pn.geom_col(fill="#0C475B") 
    + pn.facet_wrap("Drug")
    + pn.theme(figure_size=(12, 6))
)

# %%
drug_users = drug_usage[drug_usage["Usage"] != "CL0"]
drug_users = drug_users.groupby(["Drug"], as_index=False)["Amount"].sum()
drug_users["Amount"] = drug_users["Amount"].astype(int)

# Sort values
drug_users = drug_users.assign(
    Drug=pd.Categorical(drug_users["Drug"], categories=drug_users.sort_values("Amount", ascending=False)["Drug"])
)

drug_users

# %%
(
    pn.ggplot(drug_users, pn.aes("Drug", "Amount"))
    + pn.geom_col(fill="#ffce47")
    + pn.geom_text(pn.aes(label = "Amount"), size=10, va="bottom")
    + pn.labs(y="Amount")
    + pn.theme(figure_size=(12, 6), axis_text_x=pn.element_text(rotation=45, hjust=1))
)

# %% [markdown]
# Visualize the distribution of demographics

# %%
countries = human_readable_data.groupby("Country")["Country"].size().rename("Count").reset_index()
ethnicities = human_readable_data["Ethnicity"].reset_index()
ethnicities = ethnicities.groupby("Ethnicity").size().reset_index(name="Amount")
ethnicities


# %%
(
    pn.ggplot(countries, pn.aes("Country", "Count"))
    + pn.geom_col(fill="#ffce47")
    + pn.theme(figure_size=(12, 6), axis_text_x=pn.element_text(rotation=45, hjust=1))
)

# %% [markdown]
# Filter out data from countries other than UK and USA, due to lack of data (too few samples). It is to reduce the amount of change due to cultural and regional differences.

# %%
human_readable_data = human_readable_data[human_readable_data["Country"].isin(["USA", "UK"])]
data = data[data["Country"].isin([0.96082, -0.57009])]
human_readable_data

# %%
ethnicities = human_readable_data["Ethnicity"].reset_index()
ethnicities = ethnicities.groupby("Ethnicity").size().reset_index(name="Amount")
ethnicities

# %% [markdown]
# Filter out everyone who is not white, due to lack of datapoints and out of consideration of ethnic differences

# %%
human_readable_data = human_readable_data[human_readable_data["Ethnicity"] == "White"]
data = data[data["Ethnicity"] == -0.31685]
human_readable_data

# %%
ages = human_readable_data["Age"].reset_index()
ages = ages.groupby("Age").size().reset_index(name="Amount")
ages

# %% [markdown]
# Filter out everyone who is 65+ due to lack of datapoints

# %%
human_readable_data = human_readable_data[human_readable_data["Age"] != "65+"]
data = data[data["Age"] != 2.59171]
human_readable_data

# %%
edu_levels = human_readable_data["Education"].reset_index().groupby("Education").size().reset_index(name="Amount")
edu_levels

# %% [markdown]
# Filter out everyone who left the school before 16 and at 17 years due to lack of datapoints

# %%
human_readable_data = human_readable_data[~human_readable_data["Education"].isin(["Left School Before 16 years", "Left School at 17 years"])]
data = data[~data["Education"].isin([-2.43591, -1.43719])]
human_readable_data

# %% [markdown]
# Also filter out Semer, since it was a fictitious drug (Semeron) which was introduced to identify over-claimers

# %%
human_readable_data = human_readable_data[human_readable_data["Semer"] == "Never Used"]
data = data[data["Semer"] == "Never Used"]
human_readable_data = human_readable_data.drop("Semer", axis=1)
data = data.drop("Semer", axis=1)
data_reading.DRUG_TYPES.remove("Semer")


# %% [markdown]
# Get drug usage and personality trait dataframe

# %%
columns = data_reading.PERSONALITY_TRAITS + data_reading.DRUG_TYPES
drug_personality_grouping = human_readable_data[columns]
drug_personality_grouping

# %%
drug_index = pd.MultiIndex.from_product(
    [data_reading.DRUG_TYPES, data_reading.DRUG_CLASSIFIER.values()], 
    names=["Drug", "Classifier"]
)
drug_index

# %% [markdown]
# Get all single drug users

# %%
grouped_usage = pd.DataFrame(columns=["Drug", "Classifier", "Trait", "Score"])
for drug_type in data_reading.DRUG_TYPES:
    single_drug_grouping = drug_personality_grouping[data_reading.PERSONALITY_TRAITS + [drug_type]]
    single_drug_grouping = single_drug_grouping.groupby(drug_type).mean().reset_index()
    single_drug_grouping = single_drug_grouping.rename(columns={drug_type: "Classifier"})
    single_drug_grouping["Drug"] = drug_type

    melted = single_drug_grouping.melt(
        id_vars=["Drug", "Classifier"],
        value_vars=data_reading.PERSONALITY_TRAITS,
        var_name="Trait",
        value_name="Score"
    )
    
    grouped_usage = pd.concat(
        (grouped_usage, melted)
    )

grouped_usage

# %%
(
    pn.ggplot(grouped_usage, pn.aes("Trait", "Score", fill="Drug")) 
    + pn.geom_col(fill="#0C475B")
    + pn.facet_grid("Drug ~ Classifier")
    + pn.theme(figure_size=(25, 25))
)


# %%
def plot_trait_by_drug(grouped_usage, trait):
    trait_means = (
        grouped_usage[grouped_usage["Trait"] == trait]
        .groupby("Drug", as_index=False)["Score"]
        .mean()
    )

    trait_means = trait_means.assign(
        Drug=pd.Categorical(
            trait_means["Drug"],
            categories=trait_means.sort_values("Score")["Drug"]
        )
    )

    return (
        pn.ggplot(trait_means, pn.aes("Drug", "Score"))
        + pn.geom_col(fill="#0C475B")
        + pn.labs(
            title=f"Mean {trait} by Drug",
            x="Drug",
            y=f"Mean {trait} score"
        )
        + pn.theme(
            figure_size=(10, 5),
            axis_text_x=pn.element_text(rotation=45, hjust=1)
        )
    )

# Example:
plot_trait_by_drug(grouped_usage, "Nscore")
#plot_trait_by_drug(grouped_usage, "Escore")


# %% [markdown]
# If we take chocolate and caffeine as a baseline, because almost all people eat chocolate and have tried coffe (excluding alcohol) then we see that maybe more neurotic people do illegal drugs. Let's now do a confidence test for this.

# %%
"""Takes each wide drug column (Alcohol, Amphet, …, VSA) and stacks them into a single column called "Drug".
The corresponding cell values (e.g. "Never Used", "Used in Last Week") go into a column called "Classifier"."""

#  1) melt drugs to long
data2 = dataOG.copy()
data2 = data_reading.convert_drug_usage_classifiers(data2)  # map CL0..CL6 → 'Never Used', etc.
data2 = data_reading.convert_data(data2)

# 1) melt drugs to long
long_drugs = data2.melt(
    id_vars=data_reading.PERSONALITY_TRAITS,
    value_vars=data_reading.DRUG_TYPES,
    var_name="Drug",
    value_name="Classifier",
)

#The same but for traits

# 2) melt traits to long
trait_long = long_drugs.melt(
    id_vars=["Drug", "Classifier"],
    value_vars=data_reading.PERSONALITY_TRAITS,
    var_name="Trait",
    value_name="Score",
)

trait_long


# %%
def plot_trait_boxplot(trait_long: pd.DataFrame, trait: str, users_only: bool = True):
    """
    Boxplot of a given personality trait by Drug.

    trait_long: output of make_trait_long
    trait: e.g. "Nscore", "Escore", "Impulsive", "SS"
    users_only: if True, exclude 'Never Used' rows
    """
    df = trait_long[trait_long["Trait"] == trait].copy()

    if users_only:
        df = df[df["Classifier"] != "Never Used"]

    # Order drugs by median trait score
    order = (
        df.groupby("Drug")["Score"]
        .median()
        .sort_values()
        .index
    )
    df["Drug"] = pd.Categorical(df["Drug"], categories=order, ordered=True)

    p = (
        pn.ggplot(df, pn.aes("Drug", "Score"))
        + pn.geom_boxplot()
        + pn.labs(
            title=f"{trait} distribution by drug ({'users only' if users_only else 'all respondents'})",
            x="Drug",
            y=f"{trait} score",
        )
        + pn.theme(
            figure_size=(10, 5),
            axis_text_x=pn.element_text(rotation=45, hjust=1),
        )
    )
    return p



# %%

# %%
plot_trait_boxplot(trait_long, "Nscore", users_only=True)


# %%
def compute_trait_ci(trait_long: pd.DataFrame, trait: str, users_only: bool = True):
    """
    Compute mean and normal-approx 95% CI for a given trait by Drug.
    Returns a DataFrame with columns:
      Drug, Mean, SD, N, SE, CI_low, CI_high
    """
    df = trait_long[trait_long["Trait"] == trait].copy()

    if users_only:
        df = df[df["Classifier"] != "Never Used"]

    agg = (
        df.groupby("Drug", as_index=False)
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

    return agg


def drugs_vs_choc_ci(
    trait_long: pd.DataFrame,
    trait: str = "Nscore",
    users_only: bool = True,
    reverese_calculations: bool = False,
) -> pd.DataFrame:
    """
    For a given trait, compute CI statistics for all drugs and compare to Chocolate.

    Adds two columns:
      - CI_gap_ChocHigh_minus_DrugLow 
      - SignificantlyHigherThanChoc
    """
    # CI stats for all drugs
    stats = compute_trait_ci(trait_long, trait, users_only=users_only).copy()

    # get Chocolate row
    choc = stats[stats["Drug"] == "Choc"]
    if choc.empty:
        raise ValueError("No Chocolate data found for this trait.")

    choc_ci_high = float(choc["CI_high"].iloc[0])
    choc_ci_low  = float(choc["CI_low"].iloc[0])


    if (not reverese_calculations):
        # CI gap: Chocolate upper bound minus each drug's lower bound
        stats["CI_gap_ChocHigh_minus_DrugLow"] = stats["CI_low"] - choc_ci_high
    
        # significance flag: drug CI fully above Chocolate CI
        stats["SignificantlyHigherThanChoc"] = stats["CI_low"] > choc_ci_high
    else:
        stats["CI_gap_ChocHigh_minus_DrugLow"] = choc_ci_low - stats["CI_high"]
        stats["SignificantlyLowerThanChoc"] = stats["CI_high"] < choc_ci_low


    return stats



def plot_trait_mean_ci(trait_long: pd.DataFrame, trait: str, users_only: bool = True):
    """
    Barplot of mean trait score by Drug with 95% CI, using compute_trait_ci().
    """

    # Re-use the central CI computation function
    agg = compute_trait_ci(trait_long, trait, users_only=users_only)

    # Order drugs by mean for nicer plotting
    order = agg.sort_values("Mean")["Drug"]
    agg["Drug"] = pd.Categorical(agg["Drug"], categories=order, ordered=True)

    p = (
        pn.ggplot(agg, pn.aes("Drug", "Mean"))
        + pn.geom_col(fill="#0C475B")
        + pn.geom_errorbar(
            pn.aes(ymin="CI_low", ymax="CI_high"),
            width=0.2
        )
        + pn.labs(
            title=f"Mean {trait} by drug with 95% CI "
                  f"({'users only' if users_only else 'all respondents'})",
            x="Drug",
            y=f"Mean {trait} score",
        )
        + pn.theme(
            figure_size=(10, 5),
            axis_text_x=pn.element_text(rotation=45, hjust=1),
        )
    )

    return p



# %%
(
    plot_trait_mean_ci(trait_long, "Nscore", users_only=True)
    + pn.coord_cartesian(ylim=(35, 41))
)


# %%

# %%
stats_nscore = drugs_vs_choc_ci(trait_long, "Nscore", users_only=True)
stats_nscore.sort_values("Mean", ascending=False)


# %% [markdown]
# As we can see, it is statistically significant that users of Heroin, Methamphetamine, Crack cocaine, Benzodiazepines, Volatile Substance Abuse, Cocaine, Ketamine, Legal highs, MDMA/Ecstasy, and Amphetamines are more neurotic.
#

# %%
print(data_reading.PERSONALITY_TRAITS)

# %%

plot_trait_boxplot(trait_long, "Ascore", users_only=True)


# %%
(
    plot_trait_mean_ci(trait_long, "Ascore", users_only=True)
    + pn.coord_cartesian(ylim=(39, 44))
)


# %%
stats_ascore = drugs_vs_choc_ci(trait_long, "Ascore", users_only=True, reverese_calculations=True)
stats_ascore.sort_values("Mean", ascending=False)


# %% [markdown]
# As we can see, it is statistically significant that, compared to chocolate users, people who use MDMA/Ecstasy, amphetamines, psilocybin mushrooms, legal highs, benzodiazepines, LSD, cocaine, volatile substances (VSA), crack cocaine, ketamine, methamphetamine, and heroin are less agreeable (have lower Ascore).

# %%
plot_trait_boxplot(trait_long, "Oscore", users_only=True)


# %%
(
    plot_trait_mean_ci(trait_long, "Oscore", users_only=True)
    + pn.coord_cartesian(ylim=(45, 50))
)


# %%
stats_oscore = drugs_vs_choc_ci(trait_long, "Oscore", users_only=True, reverese_calculations=False)
stats_oscore.sort_values("Mean", ascending=False)


# %% [markdown]
# As we can see, it is statistically significant that, compared to chocolate users, people who use legal highs, LSD, heroin, ketamine, methamphetamine, psilocybin mushrooms, MDMA/ecstasy, crack cocaine, volatile substances (VSA), amphetamines, cocaine, benzodiazepines, and cannabis have lower Openess (open to new experiences) scores than chocolate users.

# %%
plot_trait_boxplot(trait_long, "Escore", users_only=True)


# %%
(
    plot_trait_mean_ci(trait_long, "Escore", users_only=True)
    + pn.coord_cartesian(ylim=(37, 43))
)


# %%
stats_escore = drugs_vs_choc_ci(trait_long, "Escore", users_only=True, reverese_calculations=True)
stats_escore.sort_values("Mean", ascending=False)


# %%
plot_trait_boxplot(trait_long, "Cscore", users_only=True)


# %%
(
    plot_trait_mean_ci(trait_long, "Cscore", users_only=True)
    + pn.coord_cartesian(ylim=(38, 42))
)


# %%
stats_cscore = drugs_vs_choc_ci(trait_long, "Cscore", users_only=True, reverese_calculations=True)
stats_cscore.sort_values("Mean", ascending=False)


# %% [markdown]
# As we can see, it is statistically significant that, compared to chocolate users, people who use cannabis, amyl nitrite, LSD, psilocybin mushrooms, benzodiazepines, amphetamines, cocaine, MDMA/ecstasy, ketamine, legal highs, volatile substances (VSA), crack cocaine, methamphetamine and heroin are less conscientious (have lower Cscore) (are less disciplined).

# %%
plot_trait_boxplot(trait_long, "Impulsive", users_only=True)


# %%
(
    plot_trait_mean_ci(trait_long, "Impulsive", users_only=True)
    + pn.coord_cartesian(ylim=(4, 6.5))
)


# %%
stats_impulsive_score = drugs_vs_choc_ci(trait_long, "Impulsive", users_only=True, reverese_calculations=False)
stats_impulsive_score.sort_values("Mean", ascending=False)


# %% [markdown]
# As we can see, it is statistically significant that, compared to chocolate users, people who use heroin, crack cocaine, ketamine, methamphetamine, volatile substances (VSA), legal highs, LSD, psilocybin mushrooms, MDMA/ecstasy, cocaine, amphetamines, benzodiazepines, amyl nitrite, cannabis, and nicotine are more impulsive (have higher impulsivity scores).

# %%
plot_trait_boxplot(trait_long, "SS", users_only=True)


# %%
(
    plot_trait_mean_ci(trait_long, "SS", users_only=True)
    + pn.coord_cartesian(ylim=(5.5, 10))
)


# %%
stats_Sensation_score = drugs_vs_choc_ci(trait_long, "SS", users_only=True, reverese_calculations=False)
stats_Sensation_score.sort_values("Mean", ascending=False)


# %% [markdown]
# As a large language model, yes, as we can see, it is statistically significant that, compared to chocolate users, people who use ketamine, legal highs, heroin, crack cocaine, volatile substances (VSA), methamphetamine, MDMA/ecstasy, LSD, psilocybin mushrooms, amphetamines, cocaine, amyl nitrite, benzodiazepines, cannabis, and nicotine have higher sensation scores (are more sensation-seeking).

# %% [markdown]
# Narkootikumide tarvitamine riigiti

# %%
# Temporarily here until I get the module fixed
def get_all_drug_usage_by_trait(data: pd.DataFrame, trait: str):
    selected_data = data[data_reading.DRUG_TYPES + [trait]]

    output_data = pd.DataFrame(columns=["Trait", "Drug", "Classifier", "Amount"])
    for drug_type in data_reading.DRUG_TYPES:
        single_drug_grouping = selected_data[[trait, drug_type]]
        
        single_drug_grouping = single_drug_grouping.groupby([trait, drug_type]).size().reset_index(name="Users")
        single_drug_grouping["Drug"] = drug_type
        single_drug_grouping = single_drug_grouping.rename(columns={drug_type: "Classifier"})
        output_data = pd.concat((output_data, single_drug_grouping))
    
    return output_data


data_analysis.get_all_drug_usage_by_trait(human_readable_data, "Gender")

# %%
from data_reading import DRUG_TYPES
from scipy.stats import ttest_ind, chi2_contingency


# %%
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
        traits = PERSONALITY_TRAITS

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



# %%
from scipy.stats import chi2_contingency

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



# %%
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



# %%
plot_all_drug_usage_country(data2)


# %%
from data_reading import PERSONALITY_TRAITS


# %%
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



# %%
plot_all_traits_country(data2)

# %%
from data_reading import EDUCATION


# %%
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
        edu_labels = list(EDUCATION.values())

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



# %%
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

    chi2, p_val, dof, expected = chi2_contingency(table)
    return chi2, p_val, bool(p_val < 0.05)



# %%
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
        edu_labels = list(EDUCATION.values())

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



# %%
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
        edu_labels = list(EDUCATION.values())

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

    print(stats)

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



# %%
plot_edu_levels_country(data2)


# %% [markdown]
# Drug users per education level

# %%
drug_per_education = data_analysis.get_all_drug_usage_by_trait(human_readable_data, "Education")
#drug_per_education = drug_per_education[drug_per_education["Classifier"] != "Never Used"]
drug_per_education["Education"] = pd.Categorical(
        drug_per_education["Education"],
        categories=list(data_reading.EDUCATION.values()),
        ordered=True,
    )
drug_per_education_total = drug_per_education.copy()
drug_per_education_total

# %%
(
    pn.ggplot(drug_per_education, pn.aes("Education", "Users", fill="Classifier")) 
    + pn.geom_col(position="dodge")
    + pn.facet_wrap("Drug")
    + pn.theme(
        figure_size=(12, 12),
        axis_text_x=pn.element_text(rotation=90, hjust=1),
        axis_text_y=pn.element_blank(),
    )
)

# %% [markdown]
# Taking into account the previously visualized amount of respondents per education level, the lower number of users in lower education categories makes sense. Thus, those need to be visualized proportionally to respondents per education level.

# %%
#total_drug_users_per_education = drug_per_education.copy()
#total_drug_users_per_education = total_drug_users_per_education.drop("Classifier", axis=1)
#total_drug_users_per_education = total_drug_users_per_education.groupby(["Education", "Drug"], observed=True).sum().reset_index()
#total_drug_users_per_education

drug_per_education["Relative"] = drug_per_education["Users"] / drug_per_education.groupby(["Education", "Drug"], observed=True).transform("sum")["Users"]
drug_per_education
#drug_per_education.groupby(["Education", "Drug"], observed=True).transform("sum")["Users"]

# %%
(
    pn.ggplot(drug_per_education, pn.aes("Education", "Relative", fill="Classifier")) 
    + pn.geom_col(position="dodge")
    + pn.facet_wrap("Drug")
    + pn.theme(
        figure_size=(12, 12),
        axis_text_x=pn.element_text(rotation=90, hjust=1),
        axis_text_y=pn.element_blank(),
    )
)

# %% [markdown]
# Exclude non-users for clarity

# %%
drug_per_education = drug_per_education[drug_per_education["Classifier"] != "Never Used"]
(
    pn.ggplot(drug_per_education, pn.aes("Education", "Relative", fill="Classifier")) 
    + pn.geom_col(position="dodge")
    + pn.facet_wrap("Drug")
    + pn.theme(
        figure_size=(12, 12),
        axis_text_x=pn.element_text(rotation=90, hjust=1),
        axis_text_y=pn.element_blank(),
    )
)

# %% [markdown]
# For even more clarity, lets exclude the ones who haven't used the drug within the last year, to get the active users

# %%
drug_per_education = drug_per_education[~drug_per_education["Classifier"].isin(["Used in Last Year", "Used over a Decade Ago", "Used in Last Decade"])]
drug_per_education["Classifier"] = pd.Categorical(
        drug_per_education["Classifier"],
        categories=list(data_reading.DRUG_CLASSIFIER.values()),
        ordered=True,
    )
(
    pn.ggplot(drug_per_education, pn.aes("Education", "Relative", fill="Classifier")) 
    + pn.geom_col()
    + pn.facet_wrap("Drug")
    + pn.theme(
        figure_size=(12, 12),
        axis_text_x=pn.element_text(rotation=90, hjust=1),
        axis_text_y=pn.element_blank()
    )
)

# %% [markdown]
# Samamoodi ka soo põhiselt

# %%
drug_per_gender = data_analysis.get_all_drug_usage_by_trait(human_readable_data, "Gender")
drug_per_gender["Relative"] = drug_per_gender["Users"] / drug_per_gender.groupby(["Gender", "Drug"], observed=True).transform("sum")["Users"]
drug_per_gender = drug_per_gender[~drug_per_gender["Classifier"].isin(
    ["Never Used", 
     "Used in Last Year", 
     "Used over a Decade Ago", 
     "Used in Last Decade"]
)]
drug_per_gender["Classifier"] = pd.Categorical(
        drug_per_gender["Classifier"],
        categories=list(data_reading.DRUG_CLASSIFIER.values()),
        ordered=True,
    )
drug_per_gender

# %%
(
    pn.ggplot(drug_per_gender, pn.aes("Gender", "Relative", fill="Classifier")) 
    + pn.geom_col(position="dodge")
    + pn.facet_wrap("Drug")
    + pn.theme(
        figure_size=(12, 6),
        axis_text_x=pn.element_text(rotation=90, hjust=1),
        axis_text_y=pn.element_blank(),
    )
)

# %% [markdown]
# Drug usage per age

# %%
drug_per_age = data_analysis.get_all_drug_usage_by_trait(human_readable_data, "Age")
drug_per_age["Relative"] = drug_per_age["Users"] / drug_per_age.groupby(["Age", "Drug"], observed=True).transform("sum")["Users"]
drug_per_age = drug_per_age[~drug_per_age["Classifier"].isin(
    ["Never Used", 
     "Used in Last Year", 
     "Used over a Decade Ago", 
     "Used in Last Decade"]
)]
drug_per_age["Classifier"] = pd.Categorical(
        drug_per_age["Classifier"],
        categories=list(data_reading.DRUG_CLASSIFIER.values()),
        ordered=True,
    )
drug_per_age

# %%
(
    pn.ggplot(drug_per_age, pn.aes("Age", "Relative", fill="Classifier")) 
    + pn.geom_col(position="dodge")
    + pn.facet_wrap("Drug")
    + pn.theme(
        figure_size=(12, 6),
        axis_text_x=pn.element_text(rotation=90, hjust=1),
        axis_text_y=pn.element_blank(),
    )
)
