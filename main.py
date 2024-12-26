'''
Credit card fraud detection
Author: Henry Ha
'''
# Import libraries
import pandas as pd

#TODO EDA

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Display dataset structure
print(df.info())
print(df.describe())

import matplotlib.pyplot as plt

# Plot the distribution of the target variable
class_counts = df['Class'].value_counts()
plt.figure(figsize=(6, 4))
class_counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.title("Distribution of Fraudulent and Non-Fraudulent Transactions")
plt.xticks([0, 1], ['Non-Fraudulent (0)', 'Fraudulent (1)'], rotation=0)
plt.ylabel("Number of Transactions")
plt.xlabel("Class")

# Annotate the plots with the values
for p in plt.gca().patches:
    plt.gca().text(p.get_x() + p.get_width() / 2, p.get_y() + p.get_height() / 2,
                   '{:,.0f}'.format(p.get_height()), ha="center")

plt.show()

# Plot histograms for 'Amount' and 'Time'
df[['Amount', 'Time']].hist(bins=30, figsize=(12, 5), color='lightblue', edgecolor='black')
plt.suptitle("Distribution of 'Amount' and 'Time'")
plt.show()

import seaborn as sns

# Compute correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap with annotations
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.2, annot_kws={'size': 6})
plt.title("Feature Correlation Heatmap")
plt.show()

# Boxplot for 'Amount' by 'Class'
plt.figure(figsize=(8, 6))
sns.boxplot(x='Class', y='Amount', data=df, palette='coolwarm')
plt.title("Transaction Amount Distribution by Class")
plt.show()

# Boxplot for 'Time' by 'Class'
plt.figure(figsize=(8, 6))
sns.boxplot(x='Class', y='Time', data=df, palette='coolwarm')
plt.title("Transaction Time Distribution by Class")
plt.show()

#TODO Data Wrangling and Feature Selection

from imblearn.over_sampling import SMOTE

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the new class distribution
print("Class Distribution After SMOTE:")
print(y_resampled.value_counts())

from sklearn.preprocessing import RobustScaler

# Apply RobustScaler to scale 'Amount' and 'Time'
scaler = RobustScaler()
df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])

# Compute correlation with the target variable
correlations = df.corr()['Class'].sort_values(ascending=False)
print("Correlation with Class:")
print(correlations)

from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest to compute feature importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_resampled, y_resampled)

# Extract feature importance
importances = rf.feature_importances_
features = X.columns

# Display feature importance
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance.head(10))
