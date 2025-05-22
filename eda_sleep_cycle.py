# Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style for consistency
plt.style.use('ggplot')
sns.set(font_scale=1.1)

#load the dataset
df = pd.read_csv('sleep_cycle_productivity.csv')

#Inspect : shape, types, preview
print("=== First 5 rows of the data ===")
print(df.head())

print("\nDataframe shape:", df.shape)
print("\nData types:\n", df.dtypes)

# check for missing values
print("\nmissing values per column:")
print(df.isnull().sum())

#check for duplicates
duplicates = df.duplicated().sum()
print(f"\nnumber of duplicate rows: {duplicates}")

# check for incorrect data types or obvious anomalies
print("\nsummary statistics (numerical columns):")
print(df.describe())

# if any columns are stored as strings but should be numeric, convert :
# eg:
# df['column_name'] = pd.to_numeric(df['column_name'], errors='coerce')

# value counts for categorical features (if any)
cat_columns = df.select_dtypes(include=['object', 'category']).columns
for col in cat_columns:
    print(f"\nvalue counts for {col}:")
    print(df[col].value_counts())

# correlation matrix for numerical features
print("\ncorrelation matrix:")
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)

# visualizations
os.makedirs("figures", exist_ok=True)

# histogram of mood score
plt.figure(figsize=(8,5))
sns.histplot(df['Mood Score'], bins=20, kde=True)
plt.title('Distribution of Mood Score')
plt.xlabel('Mood Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("figures/hist_mood_score.png")
plt.close()

# boxplot of Sleep Hours by Mood (if mood is categorical, otherwise skip)
# Here, assuming Mood Score is continuous, so we’ll bin it:
df['Mood_bin'] = pd.qcut(df['Mood Score'], 4, labels=["Low", "Medium", "High", "Very High"])
plt.figure(figsize=(8,5))
sns.boxplot(x='Mood_bin', y='Total Sleep Hours', data=df)
plt.title('Total Sleep Hours by Mood Quartile')
plt.xlabel('Mood Score Bin')
plt.ylabel('Total Sleep Hours')
plt.tight_layout()
plt.savefig("figures/box_sleep_by_mood.png")
plt.close()

# scatterplot: Sleep Quality vs. Mood Score
plt.figure(figsize=(8,5))
sns.scatterplot(x='Sleep Quality', y='Mood Score', data=df, alpha=0.6)
plt.title('Sleep Quality vs. Mood Score')
plt.xlabel('Sleep Quality')
plt.ylabel('Mood Score')
plt.tight_layout()
plt.savefig("figures/scatter_sleep_quality_vs_mood.png")
plt.close()

#scatterplot: Stress Level vs. Mood Score
plt.figure(figsize=(8,5))
sns.scatterplot(x='Stress Level', y='Mood Score', data=df, alpha=0.6)
plt.title('Stress Level vs. Mood Score')
plt.xlabel('Stress Level')
plt.ylabel('Mood Score')
plt.tight_layout()
plt.savefig("figures/scatter_stress_vs_mood.png")
plt.close()

#pairplot for key variables
# Select important columns for pairplot
pairplot_cols = ['Mood Score', 'Total Sleep Hours', 'Sleep Quality', 'Stress Level', 'Productivity Score']
sns.pairplot(df[pairplot_cols].dropna())
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.savefig("figures/pairplot.png")
plt.close()

#Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("figures/corr_heatmap.png")
plt.close()

#check for bias in demographic columns (e.g., Age, Gender)
if 'Gender' in df.columns:
    print("\nGender distribution:")
    print(df['Gender'].value_counts())
    plt.figure(figsize=(6,4))
    sns.countplot(x='Gender', data=df)
    plt.title('Gender Distribution')
    plt.tight_layout()
    plt.savefig("figures/gender_distribution.png")
    plt.close()

#age distribution
if 'Age' in df.columns:
    print("\nAge statistics:")
    print(df['Age'].describe())
    plt.figure(figsize=(8,5))
    sns.histplot(df['Age'], bins=20, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("figures/age_distribution.png")
    plt.close()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
'''
1. Define the Objective
- Goal: Build a regression model to predict Mood Score (1–10) from 10 input features.
- Inputs (10 features): Age, Sleep Start Time, Total Sleep Hours, Sleep Quality, Exercise (mins/day), Caffeine Intake (mg), Screen Time Before Bed (mins), Work Hours (hrs/day), Productivity Score, Stress Level.
- Output (label): Mood Score.
'''

'''
2. Data Collection & Loading
- Gather the dataset (CSV or database).
- Load into a DataFrame (e.g., pandas) and inspect the first few rows.
'''
print("\n=== New Begin ===")
df_1 = pd.read_csv('sleep_cycle_productivity.csv')

'''
3. Data Understanding & Quality Checks
- Data types: ensure all columns have appropriate types (int, float, category).
- Missing values: identify any NULLs or NaNs in the features or label.
- Outliers: detect outliers in numeric features (e.g., values beyond min/max ranges).
- Duplicates: check for and remove duplicate rows.
'''
# DATA TYPES
print("\n=== Dataframe shape ===")
print(df_1.shape)
print("\n=== Data types ===")
print(df_1.dtypes)

# MISSING VALUES
print("\n=== Missing values per column ===")
print(df_1.isnull().sum())

# OUTLIERS
# 1. Re-verify domain bounds
print("\n=== Re-verify domain bounds ===")
for col, (lo, hi) in {
    'Age': (18,60), 'Sleep Start Time': (20,24),
    'Total Sleep Hours': (4.5,9.5), 'Sleep Quality': (1,10),
    'Exercise (mins/day)': (0,89), 'Caffeine Intake (mg)': (0,299),
    'Screen Time Before Bed (mins)': (0,179),
    'Work Hours (hrs/day)': (4,12),
    'Productivity Score': (1,10), 'Stress Level': (1,10)
}.items():
    bad = (~df_1[col].between(lo,hi)).sum()
    print(f"{col}: {bad} rows outside [{lo},{hi}]")

# 2. Statistical Z-Score Audit
from scipy.stats import zscore

# Select numeric columns (excluding the label for now)
numeric_features = ['Age', 'Sleep Start Time', 'Total Sleep Hours', 'Sleep Quality', 'Exercise (mins/day)',
    'Caffeine Intake (mg)', 'Screen Time Before Bed (mins)', 'Work Hours (hrs/day)', 'Productivity Score', 'Stress Level']

print("\n=== Statistical Z-Score Audit ===")
# Compute z-scores for just the numeric features
z = df_1[numeric_features].apply(zscore)

# Count how many “extreme” values per column (|z| > 3 is a common choice)
extreme_counts = (z.abs() > 3).sum()
print("\n=== Extreme outliers per feature (|z|>3) ===")
print(extreme_counts)

# 3. Percentile-Based Inspection
print("\n=== Percentile-Based Inspection ===")
for q in [0.005, 0.01, 0.99, 0.995]:
    pct = df_1[numeric_features].quantile(q)
    print(f"Quantile {q:.1%}:\n", pct, "\n")

# 4. Visual Inspection - Box-Plot
# Set up the plot grid
print("\n=== Visual Inspection - Box-Plot ===")
plt.figure(figsize=(16, 10))
for i, col in enumerate(numeric_features, 1):
    plt.subplot(3, 4, i)  # 3 rows, 4 columns
    sns.boxplot(y=df_1[col], color='skyblue')
    plt.title(col)
    plt.tight_layout()

plt.suptitle("Boxplots of Numeric Features", fontsize=16, y=1.02)
plt.show()

# Outlier analysis
'''
1. Domain Bounds: 0 rows outside any of the 10 numeric feature ranges.
2. Z-Score Test: 0 values with |z| > 3
3. Percentile-Based Outlier Analysis
    - To ensure we weren’t missing edge cases that could skew the model, we inspected the extreme 0.5%, 1%, 99%, and 99.5% percentiles of each numeric feature.
            All tail values were found to be within realistic and predefined domain limits.
    - For example, low exercise (0 mins), high screen time (179 mins), or sleep hours between 4.5 and 9.5 all align with typical human variability.
    - Conclusion: No percentile extremes indicated data entry errors or implausible values, so all records were retained.
4. Visual Checks: Boxplots show no “fliers”
'''

# DUPLICATES
duplicates = df_1.duplicated().sum()
print("\n=== Number of duplicate rows ===")
print(duplicates)

'''
4. Data Cleaning & Preprocessing
- Handle missing values:
    - Numerical: impute with mean/median or use model-based imputation.
    - Categorical (Gender): impute with mode or introduce "Unknown" category.
- Outliers treatment: clip or transform outliers, or consider robust scaling.
- Feature type conversions:
    - Convert Gender to numeric codes or one‑hot encoding.
'''

# All of this is not necessary, since:
# 1. There are no missing values or such
# 2. There are no outliers
# 3. The features are all of correct data types

'''
5. Exploratory Data Analysis (EDA)
- Univariate analysis: histograms or boxplots for each feature and the label.
- Bivariate analysis: scatterplots of each feature vs. Mood Score; compute Pearson/Spearman correlations.
- Feature distributions: check normality; consider transformations (e.g., log for skewed).
'''
label = "Mood Score"

# Understand Distributions (Univariate Analysis)
# Plot histograms
plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_features + [label], 1):
    plt.subplot(4, 3, i)
    sns.histplot(df_1[col], kde=True, bins=30, color='cornflowerblue')
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
plt.suptitle("Feature Distributions (Including Mood Score)", fontsize=16, y=1.02)
plt.show()

# Reveal Nonlinear Relationships (Feature vs. Label) (Bivariate Analysis)
# Scatterplots: each feature vs. Mood Score
plt.figure(figsize=(18, 12))
for i, col in enumerate(numeric_features, 1):
    plt.subplot(4, 3, i)
    sns.scatterplot(data=df_1, x=col, y=label, alpha=0.5, color='teal')
    sns.regplot(data=df_1, x=col, y=label, scatter=False, color='darkred', line_kws={'linewidth': 2})
    plt.title(f'{col} vs. Mood Score')
    plt.xlabel(col)
    plt.ylabel('Mood Score')
    plt.tight_layout()
plt.suptitle("Feature vs. Mood Score Relationships", fontsize=16, y=1.02)
plt.show()
'''
Flat relationships: If there’s no visual trend, the feature might have low predictive power.
'''
# Pearson Correlation with Mood Score
correlations = df_1[numeric_features + ['Mood Score']].corr(method='pearson')['Mood Score'].sort_values(ascending=False)
print("\n=== Pearson Correlation with Mood Score ===")
print(correlations)

sns.histplot(df_1['Mood Score'], bins=10, kde=True)
plt.title("Mood Score Distribution")
plt.show()

# Correlation Matrix: All features + Mood Score
plt.figure(figsize=(12, 10))
corr_matrix = df_1[numeric_features + ['Mood Score']].corr(method='pearson')

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": .75}
)
plt.title("Correlation Heatmap of Features and Mood Score", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print(df_1['Mood Score'].value_counts().sort_index())

'''
6. Feature Engineering
- Time features: convert Sleep Start Time (e.g., 23.33) into a cyclic feature:
    - sin(2π × start/24) and cos(2π × start/24).
- Interaction terms (optional): e.g., Sleep Quality × Total Sleep Hours.
Happens in other step:
( - Scaling: Standardize (zero mean, unit variance) or normalize (min–max 0–1) for distance‑based models like kNN. )
'''
df_1["Sleep_Start_sin"] = np.sin(2 * np.pi * df_1["Sleep Start Time"] / 24)
df_1["Sleep_Start_cos"] = np.cos(2 * np.pi * df_1["Sleep Start Time"] / 24)
df_1.drop("Sleep Start Time", axis=1, inplace=True)

# Interaction terms probably not going to do anything
'''
7. Feature Selection
- Correlation threshold: drop features with very low correlation to Mood Score.
'''
# If we were to do feature selection, we would select no feature at all...

'''
8. Train/Test Split
Split data into training (e.g., 70–80%) and test (20–30%) sets; use a fixed random seed.
'''
features = ['Age', 'Total Sleep Hours', 'Sleep Quality', 'Exercise (mins/day)',
            'Caffeine Intake (mg)', 'Screen Time Before Bed (mins)', 'Work Hours (hrs/day)',
            'Productivity Score', 'Stress Level', 'Sleep_Start_sin', 'Sleep_Start_cos']
X = df_1[features]
y = df_1['Mood Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
# Feature Scaling (StandardScaler)
scaler = StandardScaler()
# Fit only on training data, transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''
9. Model Selection & Baseline
- Baseline model: simple linear regression to gauge baseline performance.
- Alternative model: k‑Nearest Neighbors regressor.
Pros & Cons:
- Linear Regression: fast, interpretable coefficients, assumes linear relationships.
- kNN: nonparametric, captures nonlinear patterns, needs careful scaling and choice of k.

10. Cross-Validation & Hyperparameter Tuning
- Use k-fold cross-validation (e.g., k=5) on training data.
- Linear Regression: regularization options (Ridge, Lasso) and tune penalty term (alpha).
- kNN: tune number of neighbors k (e.g., 1–30), distance metric (Euclidean vs. Manhattan).
- Perform GridSearchCV or RandomizedSearchCV to find best hyperparameters.

11. Model Evaluation
Metrics for regression:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² score
Compare cross-validated performance of models and select the best.
'''
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Assuming X_train_scaled, X_test_scaled, y_train, y_test are already defined

# Define models and hyperparameter grids
models = {
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {}
    },
    "kNN Regressor": {
        "model": KNeighborsRegressor(),
        "params": {"n_neighbors": list(range(1, 15)), "weights": ['uniform', 'distance']}
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]}
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}
    }
}

results = {}

# Perform GridSearchCV and evaluate on test set
for name, m in models.items():
    grid = GridSearchCV(m["model"], m["params"], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_

    # Predict on test set
    y_pred = best_model.predict(X_test_scaled)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    best_params = grid.best_params_

    results[name] = {
        "Best Params": best_params,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

df_results = pd.DataFrame(results).T.sort_values(by="RMSE")
for name, metrics in results.items():
    print(f"--- Model: {name} ---")
    print(f"  Best Parameters: {metrics['Best Params']}")
    print(f"  MAE: {metrics['MAE']:.4f}")  # Format to 4 decimal places
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  R2: {metrics['R2']:.4f}")
    print("\n") # Add an empty line for better readability

'''
12. Final Model & Test Set Validation

Retrain the selected model on the full training data with optimal hyperparameters.

Evaluate on the held-out test set to estimate real-world performance.

13. Error Analysis

Plot residuals vs. predictions to check for patterns.

Identify cases with high prediction error and investigate feature values.

14. Model Interpretation & Reporting

Linear Regression: analyze coefficients to understand feature impact.

kNN: use Partial Dependence Plots if needed to illustrate nonlinear effects.

Summarize findings, strengths, and limitations in a report.
'''