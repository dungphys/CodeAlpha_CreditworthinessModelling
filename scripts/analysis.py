from pathlib import Path

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw-data" / "Credit_Risk_Data.csv"
PROCESSED_DATA_PATH = ROOT / "data" / "processed-data" / "processed_data.csv"


def missingness_univariate_analysis(
    df: pd.DataFrame,
    target_col: str,
    excluded_cols: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Test feature associations with missingness in the target column."""
    missing = df[target_col].isna().astype(int)
    results = []

    for col in df.columns:
        if col == target_col or col in excluded_cols:
            continue
        series = df[col]

        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            contingency = pd.crosstab(series, missing)
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue
            chi2, p_value, _, _ = chi2_contingency(contingency)
            n = contingency.values.sum()
            effect = (chi2 / (n * (min(contingency.shape) - 1))) ** 0.5
            results.append((col, "categorical", p_value, effect))
        else:
            observed = series[missing == 0].dropna().astype(float)
            missing_values = series[missing == 1].dropna().astype(float)
            if len(observed) == 0 or len(missing_values) == 0:
                continue
            statistic, p_value = mannwhitneyu(observed, missing_values)
            effect = 2 * statistic / (len(observed) * len(missing_values)) - 1
            results.append((col, "numeric", p_value, effect))

    output = pd.DataFrame(
        results, columns=["feature", "type", "p_value", "effect_size"]
    )
    if output.empty:
        return output

    reject, adjusted_p, _, _ = multipletests(
        output["p_value"], alpha=alpha, method="fdr_bh"
    )
    output["p_adj_bh"] = adjusted_p
    output["significant"] = reject
    return output.sort_values("p_adj_bh").reset_index(drop=True)


def target_distrib(df: pd.DataFrame, target_col: str, which: str) -> None:
    target = df[target_col].value_counts().to_frame().reset_index()
    target["percentage"] = 100 * target["count"] / len(df)
    print(f"On {which}:")
    print(target)


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lti = df["loan_to_income_ratio"]
    grade_code = df["loan_grade"].map({grade: i for i, grade in enumerate("ABCDEFG")})
    for column in ["person_income_usd", "loan_amount_usd"]:
        df[f"log_{column}"] = np.log1p(df[column])

    df["other_debt_to_income_ratio"] = (
        df["debt_to_income_ratio"] - df["loan_to_income_ratio"]
    )
    df["lpi_to_lti_ratio"] = (
        df["loan_percent_income"] / df["loan_to_income_ratio"].replace(0, np.nan)
    )
    df["high_lti"] = (lti > 0.3).astype(int)
    df["loan_grade_sep"] = (grade_code >= 3).astype(int)
    df["high_income"] = (np.log1p(df["person_income_usd"]) > 9.9).astype(int)
    df["high_lti_x_rent"] = df["high_lti"] * (
        df["person_home_ownership"] == "RENT"
    ).astype(int)
    return df


def analyze() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    print(df.info())
    print(df.head())
    print(df.drop(columns=["client_ID"]).describe(include=["object", str]).T)
    print(df.drop(columns=["client_ID"]).describe(include="number").round(3).T)

    renamed_columns = {
        "person_age": "person_age_years",
        "person_income": "person_income_usd",
        "person_emp_length": "person_emp_length_years",
        "loan_amnt": "loan_amount_usd",
        "other_debt": "other_debt_usd",
        "cb_person_cred_hist_length": "cb_person_cred_hist_length_years",
    }
    df = df.rename(columns=renamed_columns)

    null_stat = pd.DataFrame(df.isna().sum().sort_values(ascending=False)).reset_index()
    null_stat.columns = ["feature", "null_count"]
    null_stat["null_pct"] = null_stat["null_count"] / len(df) * 100
    print(null_stat[null_stat["null_count"] > 0])

    null_cols = null_stat.loc[null_stat["null_count"] > 0, "feature"]
    completeness = df.notna().mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    completeness.plot(kind="barh", color="steelblue", ax=axes[0])
    msno.matrix(
        df[null_cols],
        figsize=(12, 6),
        color=(0.5, 0.5, 0.5),
        sparkline=False,
        ax=axes[1],
    )
    axes[1].tick_params(axis="x", rotation=30, labelsize=12)
    axes[1].tick_params(axis="y", labelsize=12)
    axes[0].set_title("Data Completeness Per Column")
    axes[0].set_xlabel("Completeness Fraction (1.0 = Fully Complete)")
    plt.tight_layout()
    plt.show()

    excluded_cols = ["client_ID", "loan_status", "city_longitude", "city_latitude"]
    for column in ["loan_int_rate", "person_emp_length_years"]:
        results = missingness_univariate_analysis(df, column, excluded_cols)
        print(f"\nMissingness analysis for {column}:")
        print(results)

    features_only = df.drop(columns=["client_ID", "loan_status"], errors="ignore")
    extended_features = df.drop(columns=["client_ID"], errors="ignore")
    print("Full dataset:", df.duplicated().sum())
    print("Features only:", features_only.duplicated().sum())
    print("Extended features:", extended_features.duplicated().sum())

    target_col = "loan_status"
    target_distrib(df, target_col, "train set")
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.countplot(data=df, x=target_col, hue=target_col, stat="percent", legend=False, ax=ax)
    plt.show()

    dem_features = [
        "person_home_ownership",
        "person_emp_length_years",
        "person_income_usd",
        "person_age_years",
        "marital_status",
        "gender",
        "country",
        "employment_type",
        "education_level",
        "city",
        "state",
    ]
    fin_features = [
        feature for feature in df.columns
        if feature not in dem_features + [
            "client_ID", "loan_status", "city_longitude", "city_latitude"
        ]
    ]
    num_features = df.select_dtypes(include=["number"]).columns.tolist()
    num_features = [
        feature for feature in num_features
        if feature not in ["client_ID", "loan_status", "city_longitude", "city_latitude"]
    ]
    cat_features = df.select_dtypes(include=[str, "bool"]).columns.tolist()
    cat_features = [
        feature for feature in cat_features if feature not in ["client_ID", "loan_status"]
    ]

    cat_dem_features = [feature for feature in dem_features if feature in cat_features]
    ncols = 3
    nrows = (len(cat_dem_features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cat_dem_features):
        proportions = pd.crosstab(
            df[col], df[target_col], normalize="index", dropna=False
        ).mul(100)
        proportions.plot(kind="bar", stacked=True, ax=ax, legend=False)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Target distribution (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=35)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title=target_col, loc="upper center",
        bbox_to_anchor=(0.1, 1.02), ncol=len(labels),
    )
    for ax in axes[len(cat_dem_features):]:
        ax.set_visible(False)
    fig.suptitle("Target distribution by categorical demographics", y=1.02)
    plt.tight_layout()
    plt.show()

    num_dem_features = [feature for feature in dem_features if feature in num_features]
    ncols = 3
    nrows = (len(num_dem_features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for feature, ax in zip(num_dem_features, axes):
        sns.boxplot(x=df[target_col], y=df[feature], ax=ax)
        ax.set_title(feature)
    for ax in axes[len(num_dem_features):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()

    print(df[df["person_emp_length_years"] > 100])
    print(df[df["person_age_years"] > 100])
    print(df[df["person_emp_length_years"] >= df["person_age_years"]])
    df = df[
        (df["person_emp_length_years"] <= 100)
        & (df["person_age_years"] <= 100)
        & (df["person_emp_length_years"] < df["person_age_years"])
    ].reset_index(drop=True).copy()
    print(df.head(3))

    ncols = 3
    nrows = (len(num_dem_features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for feature, ax in zip(num_dem_features, axes):
        if feature == "person_income_usd":
            sns.violinplot(x=df[target_col], y=np.log1p(df[feature]), inner="quartiles", ax=ax)
            ax.set_title(f"log(1 + {feature})")
        else:
            sns.violinplot(x=df[target_col], y=df[feature], inner="quartiles", ax=ax)
            ax.set_title(feature)
    for ax in axes[len(num_dem_features):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()

    mixed = ["loan_term_months", "past_delinquencies"]
    cat_fin_features = [feature for feature in fin_features if feature in cat_features + mixed]
    ncols = 3
    nrows = (len(cat_fin_features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cat_fin_features):
        proportions = pd.crosstab(
            df[col], df[target_col], normalize="index", dropna=False
        ).mul(100)
        proportions.plot(kind="bar", stacked=True, ax=ax, legend=False)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Target distribution (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=35)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title=target_col, loc="upper center",
        bbox_to_anchor=(0.1, 1.02), ncol=len(labels),
    )
    for ax in axes[len(cat_fin_features):]:
        ax.set_visible(False)
    fig.suptitle("Target distribution by categorical finance", y=1.02)
    plt.tight_layout()
    plt.show()

    num_fin_features = [
        feature for feature in fin_features
        if feature in list(set(num_features) - set(mixed))
    ]
    ncols = 3
    nrows = (len(num_fin_features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for feature, ax in zip(num_fin_features, axes):
        if feature in ["loan_amount_usd", "other_debt_usd"]:
            sns.violinplot(x=df[target_col], y=np.log1p(df[feature]), inner="quartiles", ax=ax)
            ax.set_title(f"log(1 + {feature})")
        else:
            sns.violinplot(x=df[target_col], y=df[feature], inner="quartiles", ax=ax)
            ax.set_title(feature)
    for ax in axes[len(num_fin_features):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()

    corr = (
        df.select_dtypes(include="number")
        .drop(columns=["client_ID", "city_latitude", "city_longitude"], errors="ignore")
        .corr()
    )
    fig, ax = plt.subplots(figsize=(15, 14))
    sns.heatmap(
        corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=True,
        fmt=".3f", square=True, ax=ax,
    )
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    sns.scatterplot(data=df, x="loan_to_income_ratio", y="loan_percent_income", hue=target_col)
    plt.show()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ratio = df["loan_percent_income"] / df["loan_to_income_ratio"].replace(0, np.nan)
    sns.kdeplot(x=ratio, bw_adjust=0.15, multiple="stack", ax=axes[0])
    sns.kdeplot(x=ratio, hue=df[target_col], bw_adjust=0.1, multiple="fill", ax=axes[1])
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(14, 16))
    sns.scatterplot(
        data=df, y="cb_person_cred_hist_length_years", x="person_age_years",
        hue=target_col, ax=axes[0, 0],
    )
    sns.scatterplot(
        y=np.log1p(df["other_debt_usd"]), x=np.log1p(df["person_income_usd"]),
        hue=df[target_col], ax=axes[0, 1],
    )
    sns.scatterplot(
        data=df, y="debt_to_income_ratio", x="loan_to_income_ratio",
        hue=target_col, ax=axes[1, 0],
    )
    sns.scatterplot(
        data=df, y="debt_to_income_ratio", x="loan_percent_income",
        hue=target_col, ax=axes[1, 1],
    )
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    sns.scatterplot(
        x=np.log1p(df["person_income_usd"]), y=np.log1p(df["loan_amount_usd"]),
        hue=df[target_col], ax=axes[0],
    )
    sns.scatterplot(
        x=np.log1p(df["person_income_usd"]), y=df["loan_to_income_ratio"],
        hue=df[target_col], ax=axes[1],
    )
    plt.tight_layout()
    plt.show()

    redundant_cols = [
        "client_ID", "loan_percent_income", "debt_to_income_ratio", "other_debt_usd",
        "gender", "marital_status", "city", "state", "country", "employment_type",
        "education_level", "person_income_usd", "loan_amount_usd", "city_latitude",
        "city_longitude",
    ]
    df_processed = add_columns(df)
    df_processed["cb_person_default_on_file"] = (
        df_processed["cb_person_default_on_file"] == "Y"
    ).astype(int)
    df_processed = df_processed.drop(columns=redundant_cols)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")
    return df_processed


if __name__ == "__main__":
    analyze()
