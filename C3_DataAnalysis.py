# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Visualize the distribution of numeric variables
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

print("\nVisualizing Numeric Columns Distribution...")
for col in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Step 2: Visualize correlations between numeric variables
plt.figure(figsize=(10, 8))
correlation_matrix = data[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title('Correlation Matrix')
plt.show()

# Step 3: Relationship between a target variable and features (if applicable)
# Replace 'Target_Column' with the name of your target variable
if 'Target_Column' in data.columns:  # Replace with the actual column name
    sns.pairplot(data, hue='Target_Column', diag_kind='kde', corner=True)
    plt.show()
else:
    print("\nNo target column detected for pairplot visualization.")

# Step 4: Check for class imbalance in target variable (if applicable)
# Replace 'Target_Column' with the name of your target variable
if 'Target_Column' in data.columns:  # Replace with the actual column name
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Target_Column', data=data, palette='viridis')
    plt.title('Class Distribution in Target Variable')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()
