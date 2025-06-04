"""Visualization helpers used throughout the project."""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plotting aesthetics
plt.style.use('ggplot')
sns.set(font_scale=1.1)


def create_visualizations(df, corr_matrix):
    """Generate and save a suite of exploratory figures."""
    # Ensure the output directory exists
    os.makedirs("figures", exist_ok=True)

    # Histogram of the target variable
    plt.figure(figsize=(8,5))
    sns.histplot(df['Mood Score'], bins=20, kde=True)
    plt.title('Distribution of Mood Score')
    plt.xlabel('Mood Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("figures/hist_mood_score.png")
    plt.close()

    # Boxplot of sleep hours by binned mood score
    df['Mood_bin'] = pd.qcut(df['Mood Score'], 4, labels=["Low", "Medium", "High", "Very High"])
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Mood_bin', y='Total Sleep Hours', data=df)
    plt.title('Total Sleep Hours by Mood Quartile')
    plt.xlabel('Mood Score Bin')
    plt.ylabel('Total Sleep Hours')
    plt.tight_layout()
    plt.savefig("figures/box_sleep_by_mood.png")
    plt.close()

    # Scatterplot: Sleep Quality vs. Mood Score
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='Sleep Quality', y='Mood Score', data=df, alpha=0.6)
    plt.title('Sleep Quality vs. Mood Score')
    plt.xlabel('Sleep Quality')
    plt.ylabel('Mood Score')
    plt.tight_layout()
    plt.savefig("figures/scatter_sleep_quality_vs_mood.png")
    plt.close()

    # Scatterplot: Stress Level vs. Mood Score
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='Stress Level', y='Mood Score', data=df, alpha=0.6)
    plt.title('Stress Level vs. Mood Score')
    plt.xlabel('Stress Level')
    plt.ylabel('Mood Score')
    plt.tight_layout()
    plt.savefig("figures/scatter_stress_vs_mood.png")
    plt.close()

    # Pairplot for a subset of informative columns
    pairplot_cols = ['Mood Score', 'Total Sleep Hours', 'Sleep Quality', 'Stress Level', 'Productivity Score']
    sns.pairplot(df[pairplot_cols].dropna())
    plt.suptitle("Pairplot of Key Features", y=1.02)
    plt.savefig("figures/pairplot.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig("figures/corr_heatmap.png")
    plt.close()

    # Check for potential bias in demographic columns (e.g., Age, Gender)
    if 'Gender' in df.columns:
        print("\nGender distribution:")
        print(df['Gender'].value_counts())
        plt.figure(figsize=(6,4))
        sns.countplot(x='Gender', data=df)
        plt.title('Gender Distribution')
        plt.tight_layout()
        plt.savefig("figures/gender_distribution.png")
        plt.close()

    # Age distribution
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
