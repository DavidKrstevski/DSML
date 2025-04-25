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
print("first 5 rows of the data:")
print(df.head())

print("\ndataframe shape:", df.shape)
print("\ndata types:\n", df.dtypes)

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
sns.histplot(df['M´mood Score'], bins=20, kde=True)
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
