# Seasonal Agriculture Performance Analysis
# VS Code version
# Keep this Python file and seasonal_agriculture_performance_dataset.csv in the same folder.
#
# Install required packages in VS Code terminal:
# pip install pandas numpy matplotlib seaborn scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


# ============================================================
# 1. LOAD DATASET
# ============================================================

CSV_FILE = Path("seasonal_agriculture_performance_dataset.csv")

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"CSV file not found: {CSV_FILE}\n"
        "Put this Python file and the CSV file in the same folder."
    )

df_raw = pd.read_csv(CSV_FILE)
df = df_raw.copy()

print("=" * 80)
print("SEASONAL AGRICULTURE PERFORMANCE ANALYSIS")
print("=" * 80)

print("\nDataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 2. TOP 5 ROWS
# ============================================================

print("\n" + "=" * 80)
print("TOP 5 ROWS")
print("=" * 80)
print(df.head().to_string(index=False))


# ============================================================
# 3. DATASET SHAPE AND STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("DATASET SHAPE AND STRUCTURE")
print("=" * 80)

print("Shape:", df.shape)

print("\nColumn names:")
for col in df.columns:
    print("-", col)

print("\nDataset information:")
df.info()

print("\nData types:")
print(df.dtypes)


# ============================================================
# 4. CATEGORICAL VARIABLES
# ============================================================

print("\n" + "=" * 80)
print("CATEGORICAL VARIABLES")
print("=" * 80)

for col in ["State", "District", "Crop", "Season", "Irrigation_Method"]:
    if col in df.columns:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts(dropna=False))


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUE ANALYSIS")
print("=" * 80)

missing = pd.DataFrame({
    "Missing_Count": df.isna().sum(),
    "Missing_Percentage": (df.isna().mean() * 100).round(2)
})

missing = missing[missing["Missing_Count"] > 0]

if missing.empty:
    print("No missing values found.")
else:
    print(missing.sort_values("Missing_Count", ascending=False))


# ============================================================
# 6. DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("DUPLICATE RECORD ANALYSIS")
print("=" * 80)

duplicate_count = df.duplicated().sum()
print("Duplicate records:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicate records removed.")
else:
    print("No duplicate records found.")


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("HANDLING MISSING VALUES")
print("=" * 80)

# Rainfall: Season-wise median
if "Rainfall_mm" in df.columns:
    df["Rainfall_mm"] = df["Rainfall_mm"].fillna(
        df.groupby("Season")["Rainfall_mm"].transform("median")
    )
    df["Rainfall_mm"] = df["Rainfall_mm"].fillna(df["Rainfall_mm"].median())

# Soil moisture: Season-wise median
if "Soil_Moisture_pct" in df.columns:
    df["Soil_Moisture_pct"] = df["Soil_Moisture_pct"].fillna(
        df.groupby("Season")["Soil_Moisture_pct"].transform("median")
    )
    df["Soil_Moisture_pct"] = df["Soil_Moisture_pct"].fillna(
        df["Soil_Moisture_pct"].median()
    )

# Yield: Crop + Season median
if "Yield_Tonnes_Ha" in df.columns:
    df["Yield_Tonnes_Ha"] = df["Yield_Tonnes_Ha"].fillna(
        df.groupby(["Crop", "Season"])["Yield_Tonnes_Ha"].transform("median")
    )
    df["Yield_Tonnes_Ha"] = df["Yield_Tonnes_Ha"].fillna(
        df["Yield_Tonnes_Ha"].median()
    )

# Fallback for any other numeric missing values
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

print("Remaining missing values:")
remaining_missing = df.isna().sum()
print(remaining_missing[remaining_missing > 0])

print("\nTotal remaining missing values:",
      int(df.isna().sum().sum()))


# ============================================================
# 8. FINAL DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATA QUALITY CHECK")
print("=" * 80)

print("Final shape:", df.shape)
print("Total missing values:", int(df.isna().sum().sum()))
print("Total duplicate rows:", int(df.duplicated().sum()))


# ============================================================
# 9. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("DESCRIPTIVE / STATISTICAL ANALYSIS")
print("=" * 80)

numeric_cols = df.select_dtypes(include=np.number).columns

print("\nDescriptive statistics:")
print(df[numeric_cols].describe().T)


# ============================================================
# 10. SEASONAL SUMMARY
# ============================================================

season_summary = df.groupby("Season").agg(
    Farm_Count=("Farm_ID", "count"),
    Avg_Yield=("Yield_Tonnes_Ha", "mean"),
    Median_Yield=("Yield_Tonnes_Ha", "median"),
    Avg_Production=("Production_Tonnes", "mean"),
    Avg_Revenue=("Revenue_INR", "mean"),
    Avg_Cost=("Total_Cost_INR", "mean"),
    Avg_Profit=("Profit_INR", "mean"),
    Avg_Water_Used=("Water_Used_m3", "mean"),
    Avg_Water_Efficiency=("Water_Efficiency_t_per_1000m3", "mean"),
    Avg_Disease_Risk=("Disease_Pest_Risk_pct", "mean"),
    Avg_Rainfall=("Rainfall_mm", "mean"),
    Avg_Temperature=("Avg_Temperature_C", "mean")
).sort_values("Avg_Profit", ascending=False)

print("\n" + "=" * 80)
print("SEASONAL SUMMARY")
print("=" * 80)
print(season_summary.round(2))


# ============================================================
# 11. OUTLIER ANALYSIS USING IQR
# ============================================================

print("\n" + "=" * 80)
print("OUTLIER INVESTIGATION")
print("=" * 80)


def outlier_summary(data, columns):
    result = []

    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = ((data[col] < lower) | (data[col] > upper)).sum()

        result.append({
            "Column": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower_Bound": lower,
            "Upper_Bound": upper,
            "Outlier_Count": int(count),
            "Outlier_Percentage": round(
                count / len(data) * 100, 2
            )
        })

    return pd.DataFrame(result).sort_values(
        "Outlier_Count",
        ascending=False
    )


outliers = outlier_summary(df, numeric_cols)
print(outliers.round(2).to_string(index=False))

print("\nOutliers are investigated but not automatically removed because")
print("unusual values may represent genuine agricultural observations.")


# ============================================================
# 12. UNIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("UNIVARIATE ANALYSIS")
print("=" * 80)

# Season distribution
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Season",
    order=df["Season"].value_counts().index
)
plt.title("Number of Farms by Season")
plt.xlabel("Season")
plt.ylabel("Number of Farms")
plt.tight_layout()
plt.show()

# Crop distribution
plt.figure(figsize=(10, 5))
sns.countplot(
    data=df,
    x="Crop",
    order=df["Crop"].value_counts().index
)
plt.title("Number of Farms by Crop")
plt.xlabel("Crop")
plt.ylabel("Number of Farms")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Irrigation method distribution
plt.figure(figsize=(9, 5))
sns.countplot(
    data=df,
    x="Irrigation_Method",
    order=df["Irrigation_Method"].value_counts().index
)
plt.title("Irrigation Method Distribution")
plt.xlabel("Irrigation Method")
plt.ylabel("Number of Farms")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

# Numerical distributions
hist_cols = [
    "Farm_Area_Hectares",
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Soil_Moisture_pct",
    "Yield_Tonnes_Ha",
    "Profit_INR",
    "Water_Used_m3",
    "Disease_Pest_Risk_pct"
]

for col in hist_cols:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()


# ============================================================
# 13. BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("BIVARIATE ANALYSIS")
print("=" * 80)

# Season vs Yield
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Season", y="Yield_Tonnes_Ha")
plt.title("Yield Distribution Across Seasons")
plt.xlabel("Season")
plt.ylabel("Yield (Tonnes/Ha)")
plt.tight_layout()
plt.show()

# Season vs Profit
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Season", y="Profit_INR")
plt.title("Profit Distribution Across Seasons")
plt.xlabel("Season")
plt.ylabel("Profit (INR)")
plt.tight_layout()
plt.show()

# Rainfall vs Yield
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="Rainfall_mm",
    y="Yield_Tonnes_Ha",
    hue="Season",
    alpha=0.6
)
plt.title("Rainfall vs Yield")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Yield (Tonnes/Ha)")
plt.tight_layout()
plt.show()

# Fertilizer vs Yield
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="Fertilizer_kg_ha",
    y="Yield_Tonnes_Ha",
    hue="Season",
    alpha=0.6
)
plt.title("Fertilizer Usage vs Yield")
plt.xlabel("Fertilizer (kg/ha)")
plt.ylabel("Yield (Tonnes/Ha)")
plt.tight_layout()
plt.show()

# Farm area vs profit
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="Farm_Area_Hectares",
    y="Profit_INR",
    hue="Season",
    alpha=0.6
)
plt.title("Farm Area vs Profit")
plt.xlabel("Farm Area (Hectares)")
plt.ylabel("Profit (INR)")
plt.tight_layout()
plt.show()


# ============================================================
# 14. MULTIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("MULTIVARIATE ANALYSIS")
print("=" * 80)

# Crop × Season Yield
crop_season_yield = pd.pivot_table(
    df,
    values="Yield_Tonnes_Ha",
    index="Crop",
    columns="Season",
    aggfunc="mean"
)

print("\nAverage yield by crop and season:")
print(crop_season_yield.round(2))

plt.figure(figsize=(9, 6))
sns.heatmap(
    crop_season_yield,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu"
)
plt.title("Average Yield by Crop and Season")
plt.xlabel("Season")
plt.ylabel("Crop")
plt.tight_layout()
plt.show()

# State × Season Profit
state_season_profit = pd.pivot_table(
    df,
    values="Profit_INR",
    index="State",
    columns="Season",
    aggfunc="mean"
)

print("\nAverage profit by state and season:")
print(state_season_profit.round(2))

plt.figure(figsize=(10, 6))
sns.heatmap(
    state_season_profit,
    annot=True,
    fmt=".0f",
    cmap="RdYlGn",
    center=0
)
plt.title("Average Profit by State and Season")
plt.xlabel("Season")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# Pair plot using sample
pair_cols = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Soil_Moisture_pct",
    "Fertilizer_kg_ha",
    "Water_Used_m3",
    "Yield_Tonnes_Ha",
    "Profit_INR",
    "Disease_Pest_Risk_pct"
]

pair_cols = [c for c in pair_cols if c in df.columns]

sample = df[pair_cols + ["Season"]].sample(
    min(800, len(df)),
    random_state=42
)

sns.pairplot(
    sample,
    hue="Season",
    corner=True
)
plt.show()


# ============================================================
# 15. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

corr = df[numeric_cols].corr()

plt.figure(figsize=(15, 11))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.3
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

yield_corr = (
    corr["Yield_Tonnes_Ha"]
    .drop("Yield_Tonnes_Ha")
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)

profit_corr = (
    corr["Profit_INR"]
    .drop("Profit_INR")
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)

print("\nStrongest correlations with Yield:")
print(yield_corr.round(3))

print("\nStrongest correlations with Profit:")
print(profit_corr.round(3))


# ============================================================
# 16. SEASONAL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("SEASONAL COMPARISON")
print("=" * 80)

metrics = df.groupby("Season").agg(
    Average_Yield=("Yield_Tonnes_Ha", "mean"),
    Average_Production=("Production_Tonnes", "mean"),
    Average_Revenue=("Revenue_INR", "mean"),
    Average_Cost=("Total_Cost_INR", "mean"),
    Average_Profit=("Profit_INR", "mean"),
    Average_Water_Used=("Water_Used_m3", "mean"),
    Average_Water_Efficiency=("Water_Efficiency_t_per_1000m3", "mean"),
    Average_Disease_Risk=("Disease_Pest_Risk_pct", "mean"),
    Average_Rainfall=("Rainfall_mm", "mean"),
    Average_Temperature=("Avg_Temperature_C", "mean")
)

print(metrics.round(2))

for col in [
    "Average_Yield",
    "Average_Profit",
    "Average_Water_Efficiency",
    "Average_Disease_Risk"
]:
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=metrics.index,
        y=metrics[col].values
    )
    plt.title(f"{col.replace('_', ' ')} by Season")
    plt.xlabel("Season")
    plt.ylabel(col.replace("_", " "))
    plt.tight_layout()
    plt.show()


# ============================================================
# 17. STUDENT-DESIGNED ANALYSIS 1
# Seasonal Profitability
# ============================================================

print("\n" + "=" * 80)
print("STUDENT-DESIGNED ANALYSIS 1")
print("Seasonal Profitability")
print("=" * 80)

profit_analysis = df.groupby("Season").agg(
    Avg_Yield=("Yield_Tonnes_Ha", "mean"),
    Avg_Revenue=("Revenue_INR", "mean"),
    Avg_Cost=("Total_Cost_INR", "mean"),
    Avg_Profit=("Profit_INR", "mean")
)

profit_analysis["Profit_Margin_pct"] = (
    profit_analysis["Avg_Profit"] /
    profit_analysis["Avg_Revenue"] * 100
)

print(profit_analysis.round(2))


# ============================================================
# 18. STUDENT-DESIGNED ANALYSIS 2
# Irrigation Method × Season
# ============================================================

print("\n" + "=" * 80)
print("STUDENT-DESIGNED ANALYSIS 2")
print("Irrigation Method × Season")
print("=" * 80)

irrigation = df.groupby(
    ["Season", "Irrigation_Method"]
).agg(
    Avg_Yield=("Yield_Tonnes_Ha", "mean"),
    Avg_Water_Used=("Water_Used_m3", "mean"),
    Avg_Water_Efficiency=(
        "Water_Efficiency_t_per_1000m3",
        "mean"
    ),
    Avg_Profit=("Profit_INR", "mean"),
    Farm_Count=("Farm_ID", "count")
).reset_index()

print(irrigation.round(2).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(
    data=irrigation,
    x="Season",
    y="Avg_Water_Efficiency",
    hue="Irrigation_Method"
)
plt.title("Water Efficiency by Irrigation Method and Season")
plt.xlabel("Season")
plt.ylabel("Tonnes per 1,000 m³")
plt.tight_layout()
plt.show()


# ============================================================
# 19. STUDENT-DESIGNED ANALYSIS 3
# Crop × Season Performance and Risk
# ============================================================

print("\n" + "=" * 80)
print("STUDENT-DESIGNED ANALYSIS 3")
print("Crop × Season Performance and Risk")
print("=" * 80)

crop_season = df.groupby(
    ["Crop", "Season"]
).agg(
    Avg_Yield=("Yield_Tonnes_Ha", "mean"),
    Avg_Profit=("Profit_INR", "mean"),
    Avg_Disease_Risk=("Disease_Pest_Risk_pct", "mean"),
    Avg_Water_Efficiency=(
        "Water_Efficiency_t_per_1000m3",
        "mean"
    ),
    Farm_Count=("Farm_ID", "count")
).reset_index()

print("\nTop crop-season combinations by yield:")
print(
    crop_season
    .sort_values("Avg_Yield", ascending=False)
    .head(15)
    .round(2)
    .to_string(index=False)
)

yield_matrix = crop_season.pivot(
    index="Crop",
    columns="Season",
    values="Avg_Yield"
)

risk_matrix = crop_season.pivot(
    index="Crop",
    columns="Season",
    values="Avg_Disease_Risk"
)

plt.figure(figsize=(9, 6))
sns.heatmap(
    yield_matrix,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu"
)
plt.title("Crop × Season Average Yield")
plt.xlabel("Season")
plt.ylabel("Crop")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))
sns.heatmap(
    risk_matrix,
    annot=True,
    fmt=".1f",
    cmap="OrRd"
)
plt.title("Crop × Season Disease/Pest Risk")
plt.xlabel("Season")
plt.ylabel("Crop")
plt.tight_layout()
plt.show()


# ============================================================
# 20. REGIONAL SEASONAL ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("REGIONAL SEASONAL ANALYSIS")
print("=" * 80)

state_season = df.groupby(
    ["Season", "State"]
).agg(
    Avg_Yield=("Yield_Tonnes_Ha", "mean"),
    Avg_Profit=("Profit_INR", "mean"),
    Avg_Water_Efficiency=(
        "Water_Efficiency_t_per_1000m3",
        "mean"
    ),
    Farm_Count=("Farm_ID", "count")
).reset_index()

print(
    state_season
    .sort_values(
        ["Season", "Avg_Profit"],
        ascending=[True, False]
    )
    .head(20)
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 21. STATISTICAL TEST
# Kruskal-Wallis
# ============================================================

print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

test_results = []

for variable in [
    "Yield_Tonnes_Ha",
    "Profit_INR",
    "Water_Efficiency_t_per_1000m3",
    "Disease_Pest_Risk_pct"
]:
    groups = [
        group[variable].dropna()
        for _, group in df.groupby("Season")
    ]

    H, p = stats.kruskal(*groups)

    test_results.append({
        "Variable": variable,
        "Kruskal_Wallis_H": H,
        "p_value": p,
        "Significant_at_0.05":
            "Yes" if p < 0.05 else "No"
    })

stat_results = pd.DataFrame(test_results)

print(stat_results.round(4).to_string(index=False))

print(
    "\nInterpretation: p < 0.05 provides evidence that at least "
    "one season differs from another for that variable. "
    "It does not prove causation."
)


# ============================================================
# 22. 10 MEANINGFUL DATA-DRIVEN INSIGHTS
# ============================================================

print("\n" + "=" * 80)
print("10 MEANINGFUL DATA-DRIVEN INSIGHTS")
print("=" * 80)

best_yield = metrics["Average_Yield"].idxmax()
worst_yield = metrics["Average_Yield"].idxmin()

best_profit = metrics["Average_Profit"].idxmax()
worst_profit = metrics["Average_Profit"].idxmin()

best_water = metrics["Average_Water_Efficiency"].idxmax()
worst_water = metrics["Average_Water_Efficiency"].idxmin()

highest_risk = metrics["Average_Disease_Risk"].idxmax()
lowest_risk = metrics["Average_Disease_Risk"].idxmin()

top_crop = crop_season.loc[
    crop_season["Avg_Yield"].idxmax()
]

top_crop_profit = crop_season.loc[
    crop_season["Avg_Profit"].idxmax()
]

top_irrigation = irrigation.loc[
    irrigation["Avg_Water_Efficiency"].idxmax()
]

top_state = state_season.loc[
    state_season["Avg_Profit"].idxmax()
]

strongest_yield_factor = yield_corr.index[0]
strongest_yield_corr = yield_corr.iloc[0]

strongest_profit_factor = profit_corr.index[0]
strongest_profit_corr = profit_corr.iloc[0]

insights = [
    f"1. {best_yield} has the highest average yield "
    f"({metrics.loc[best_yield, 'Average_Yield']:.2f} tonnes/ha), "
    f"while {worst_yield} has the lowest "
    f"({metrics.loc[worst_yield, 'Average_Yield']:.2f}).",

    f"2. {best_profit} has the highest average profit "
    f"(INR {metrics.loc[best_profit, 'Average_Profit']:,.2f}), "
    f"while {worst_profit} has the lowest "
    f"(INR {metrics.loc[worst_profit, 'Average_Profit']:,.2f}).",

    f"3. {best_water} has the highest average water efficiency "
    f"({metrics.loc[best_water, 'Average_Water_Efficiency']:.2f} "
    f"tonnes/1,000 m³), compared with {worst_water} "
    f"({metrics.loc[worst_water, 'Average_Water_Efficiency']:.2f}).",

    f"4. Average disease/pest risk is highest in {highest_risk} "
    f"({metrics.loc[highest_risk, 'Average_Disease_Risk']:.2f}%) "
    f"and lowest in {lowest_risk} "
    f"({metrics.loc[lowest_risk, 'Average_Disease_Risk']:.2f}%).",

    f"5. The highest-yielding crop-season combination is "
    f"{top_crop['Crop']} in {top_crop['Season']} "
    f"({top_crop['Avg_Yield']:.2f} tonnes/ha).",

    f"6. The highest-profit crop-season combination is "
    f"{top_crop_profit['Crop']} in {top_crop_profit['Season']} "
    f"(INR {top_crop_profit['Avg_Profit']:,.2f}).",

    f"7. The most water-efficient irrigation-season combination "
    f"is {top_irrigation['Irrigation_Method']} in "
    f"{top_irrigation['Season']} "
    f"({top_irrigation['Avg_Water_Efficiency']:.2f} tonnes/1,000 m³).",

    f"8. The highest-profit state-season combination is "
    f"{top_state['State']} in {top_state['Season']} "
    f"(INR {top_state['Avg_Profit']:,.2f}).",

    f"9. The strongest absolute correlation with yield is "
    f"{strongest_yield_factor} (r = {strongest_yield_corr:.3f}).",

    f"10. The strongest absolute correlation with profit is "
    f"{strongest_profit_factor} (r = {strongest_profit_corr:.3f})."
]

for insight in insights:
    print(insight)

print(
    "\nNote: Correlation describes association, not causation."
)


# ============================================================
# 23. EVIDENCE-BASED RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 80)
print("EVIDENCE-BASED RECOMMENDATIONS")
print("=" * 80)

recommendations = [
    f"1. Investigate the conditions and farming practices associated "
    f"with {best_yield}, the highest-yielding season.",

    f"2. Examine the cost and revenue structure of {best_profit} "
    f"before making seasonal profitability decisions.",

    f"3. Study irrigation practices associated with {best_water} "
    f"to understand its higher observed water efficiency.",

    f"4. Increase disease/pest monitoring during {highest_risk}, "
    f"which has the highest observed average risk.",

    f"5. Investigate {top_crop['Crop']} during {top_crop['Season']} "
    f"as a promising crop-season combination based on observed yield.",

    f"6. Compare {top_irrigation['Irrigation_Method']} with other "
    f"irrigation methods using both water efficiency and profit "
    f"before recommending adoption.",

    f"7. Investigate local conditions behind the best state-season "
    f"profit combination: {top_state['State']} in {top_state['Season']}.",

    "8. Use correlation results as supporting evidence only; "
    "do not treat correlation as proof of cause and effect.",

    "9. Seasonal planning should consider productivity, profitability, "
    "water efficiency and disease/pest risk together."
]

for recommendation in recommendations:
    print(recommendation)


# ============================================================
# 24. LIMITATIONS
# ============================================================

print("\n" + "=" * 80)
print("LIMITATIONS")
print("=" * 80)

limitations = [
    "1. The analysis uses only variables available in the supplied dataset.",
    "2. Observational data can identify associations but cannot establish causation.",
    "3. Missing values were imputed using contextual medians, which may introduce bias.",
    "4. Outliers were investigated but not automatically removed because they may be genuine observations.",
    "5. Seasonal comparisons may be affected by crop, state, farm size, irrigation and other factors.",
    "6. Market prices and economic outcomes can vary by location and time.",
    "7. Some important real-world agricultural factors may not be represented.",
    "8. Statistical significance should be considered together with practical importance."
]

for limitation in limitations:
    print(limitation)


# ============================================================
# 25. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 80)
print("FINAL CONCLUSION")
print("=" * 80)

print(
    f"The analysis examined {len(df):,} farm records and "
    f"{df.shape[1]} variables. Agricultural performance varies "
    f"across seasons in terms of yield, profit, water efficiency "
    f"and disease/pest risk."
)

print(
    f"{best_yield} has the highest observed average yield, while "
    f"{best_profit} has the highest observed average profit."
)

print(
    f"{best_water} has the highest observed water efficiency, "
    f"and {highest_risk} has the highest observed disease/pest risk."
)

print(
    "Crop-season, irrigation-season and state-season comparisons "
    "show that agricultural performance is not uniform across groups."
)

print(
    "Overall, the findings can support evidence-based seasonal "
    "agricultural planning. However, the results represent patterns "
    "in this dataset and should not be interpreted as proof of "
    "causal relationships."
)


# ============================================================
# 26. FINAL CHECKLIST
# ============================================================

print("\n" + "=" * 80)
print("FINAL SUBMISSION CHECKLIST")
print("=" * 80)

checklist = [
    "Dataset loaded successfully",
    "Top 5 rows analyzed",
    "Dataset shape and structure examined",
    "Data types examined",
    "Missing values identified and handled",
    "Duplicate records identified and handled",
    "Descriptive/statistical analysis performed",
    "Outliers investigated",
    "Univariate analysis completed",
    "Bivariate analysis completed",
    "Multivariate analysis completed",
    "Correlation analysis completed",
    "Seasonal comparisons performed",
    "At least 3 student-designed analyses completed",
    "At least 8 meaningful insights documented",
    "Evidence-based recommendations provided",
    "Limitations discussed",
    "Final conclusion provided"
]

for item in checklist:
    print("[✓]", item)

print("\nAnalysis completed successfully.")
print("Check all graphs and tables before submitting the project.")
