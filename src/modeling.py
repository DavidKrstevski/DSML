"""Model training and evaluation utilities."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import zscore


# Acceptable bounds for each numeric feature used to audit the data
DOMAIN_BOUNDS = {
    'Age': (18, 60),
    'Sleep Start Time': (20, 24),
    'Total Sleep Hours': (4.5, 9.5),
    'Sleep Quality': (1, 10),
    'Exercise (mins/day)': (0, 89),
    'Caffeine Intake (mg)': (0, 299),
    'Screen Time Before Bed (mins)': (0, 179),
    'Work Hours (hrs/day)': (4, 12),
    'Productivity Score': (1, 10),
    'Stress Level': (1, 10),
}


def run_full_analysis(df):
    """Execute the entire analysis pipeline from the original script."""
    print("\n=== New Begin ===")
    # Work on a copy so the original data remains unchanged
    df_1 = df.copy()

    # -- DATA TYPES --
    print("\n=== Dataframe shape ===")
    print(df_1.shape)
    print("\n=== Data types ===")
    print(df_1.dtypes)

    # -- MISSING VALUES --
    print("\n=== Missing values per column ===")
    print(df_1.isnull().sum())

    # -- OUTLIERS --
    # 1. Re-verify domain bounds
    print("\n=== Re-verify domain bounds ===")
    for col, (lo, hi) in DOMAIN_BOUNDS.items():
        bad = (~df_1[col].between(lo,hi)).sum()
        print(f"{col}: {bad} rows outside [{lo},{hi}]")

    # 2. Statistical Z-Score Audit
    numeric_features = ['Age', 'Sleep Start Time', 'Total Sleep Hours', 'Sleep Quality',
        'Exercise (mins/day)', 'Caffeine Intake (mg)', 'Screen Time Before Bed (mins)',
        'Work Hours (hrs/day)', 'Productivity Score', 'Stress Level']

    print("\n=== Statistical Z-Score Audit ===")
    z = df_1[numeric_features].apply(zscore)
    extreme_counts = (z.abs() > 3).sum()
    print("\n=== Extreme outliers per feature (|z|>3) ===")
    print(extreme_counts)

    # 3. Percentile-Based Inspection
    print("\n=== Percentile-Based Inspection ===")
    for q in [0.005, 0.01, 0.99, 0.995]:
        pct = df_1[numeric_features].quantile(q)
        print(f"Quantile {q:.1%}:\n", pct, "\n")

    # 4. Visual Inspection - Box-Plot
    print("\n=== Visual Inspection - Box-Plot ===")
    plt.figure(figsize=(16, 10))
    for i, col in enumerate(numeric_features, 1):
        plt.subplot(3, 4, i)
        sns.boxplot(y=df_1[col], color='skyblue')
        plt.title(col)
        plt.tight_layout()
    plt.suptitle("Boxplots of Numeric Features", fontsize=16, y=1.02)
    plt.show()

    # Outlier analysis comments kept from original script

    # -- DUPLICATES --
    duplicates = df_1.duplicated().sum()
    print("\n=== Number of duplicate rows ===")
    print(duplicates)

    # All of this is not necessary, since:
    # 1. There are no missing values or such
    # 2. There are no outliers
    # 3. The features are all of correct data types

    # -- Exploratory Data Analysis (EDA) --
    label = "Mood Score"
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

    correlations = df_1[numeric_features + ['Mood Score']].corr(method='pearson')['Mood Score'].sort_values(ascending=False)
    print("\n=== Pearson Correlation with Mood Score ===")
    print(correlations)

    sns.histplot(df_1['Mood Score'], bins=10, kde=True)
    plt.title("Mood Score Distribution")
    plt.show()

    plt.figure(figsize=(12, 10))
    corr_matrix = df_1[numeric_features + ['Mood Score']].corr(method='pearson')
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True, cbar_kws={"shrink": .75})
    plt.title("Correlation Heatmap of Features and Mood Score", fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print(df_1['Mood Score'].value_counts().sort_index())

    sns.pairplot(df_1[['Mood Score', 'Total Sleep Hours', 'Work Hours (hrs/day)',
                       'Caffeine Intake (mg)', 'Stress Level']])
    plt.suptitle("Pairplot of Key Features", y=1.02)
    plt.show()

    # -- Feature Engineering --
    df_1["Sleep_Start_sin"] = np.sin(2 * np.pi * df_1["Sleep Start Time"] / 24)
    df_1["Sleep_Start_cos"] = np.cos(2 * np.pi * df_1["Sleep Start Time"] / 24)
    df_1.drop("Sleep Start Time", axis=1, inplace=True)

    # If we were to do feature selection, we would select no feature at all...

    # -- Train/Test Split --
    features = ['Age', 'Total Sleep Hours', 'Sleep Quality', 'Exercise (mins/day)',
                'Caffeine Intake (mg)', 'Screen Time Before Bed (mins)', 'Work Hours (hrs/day)',
                'Productivity Score', 'Stress Level', 'Sleep_Start_sin', 'Sleep_Start_cos']
    X = df_1[features]
    y = df_1['Mood Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA trial (not used in final model)
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_train_scaled)
    explained = pca.explained_variance_ratio_
    print(f"\nExplained variance by first 5 PCA components: {explained}")

    # -- Model Selection & Baseline --
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

    # Track metrics for each candidate model
    results = {}

    for name, m in models.items():
        grid = GridSearchCV(m["model"], m["params"], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
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
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  R2: {metrics['R2']:.4f}")
        print("\n")

    # -- Final Model & Test Set Validation --
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Residual analysis
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, linestyle='--', color='red')
    plt.xlabel('Predicted Mood Score')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Mood Score')
    plt.tight_layout()
    plt.show()

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature Importances:\n", importances)

    # Remaining commentary from the script preserved below
    print(
        """
-------------------------------
Interpretation and Discussion
-------------------------------

TECHNICAL DEPLOYMENT:
----------------------
The trained model can be deployed via a RESTful API using Flask or FastAPI.
It would expose an endpoint like /predict, accept JSON input with preprocessed features,
and return the predicted Mood Score.
For full deployment, you can use Docker to containerize the app and deploy to AWS or Azure.

DOMAIN INSIGHT:
----------------
Although individual correlations with Mood Score are weak, Gradient Boosting captures
nonlinear patterns.
Features like 'Work Hours', 'Sleep Hours', 'Exercise' and 'Caffeine Intake' had
the highest importance.
These align well with intuition — physical and lifestyle patterns moderately influence mood.

SOCIETAL REFLECTION:
---------------------
A model like this could be used in health tech or workplace wellness to proactively support mental health.
However, using mood prediction at scale raises ethical concerns:
 - Privacy: Personal sleep and health data is sensitive.
 - Misuse: Employers or insurers may misuse predictions to make decisions about individuals.
 - Fairness: Mood is influenced by socio-cultural factors not present in the data.

Therefore, deployment must ensure transparency, user consent, and fairness monitoring.

FINAL CONCLUSION: Why This Dataset Fails for Predictive Modeling
---------------------
Despite following a full machine learning pipeline — including thorough data cleaning, scaling, feature engineering,
model selection, and evaluation — the results suggest that this dataset is not suitable for reliable prediction of Mood Score
using machine learning.

Here's why, from a technical standpoint:

1. Extremely Low Feature-Label Correlation
The Pearson correlation coefficients between all features and the target (Mood Score) are close to zero (mostly in the range of −0.02 to +0.02).

This means there is no linear relationship between individual input features and the output.

Even complex models like Gradient Boosting, which can learn nonlinear patterns, failed to extract strong signals — their R² score is very low (often near 0.0), meaning the model explains almost none of the variance in the data.

2. Uniform Target Distribution
The Mood Score is evenly distributed between 1 and 10.

While this looks ideal from a balance standpoint, the problem is that the model never learns to favor one score over another — because there is no distinguishable feature profile associated with any specific mood.

3. Lack of Feature Variance Impact on Target
Even with transformations like:
 -Standardization
 -Complex interactions via Gradient Boosting

...the model fails to improve beyond a baseline predictor.

PCA further confirms this — the principal components do not capture any strong structure that aligns with Mood Score.
        """
    )
