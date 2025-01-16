# Step 1: Handle Missing Values
# Replace missing numeric values with the mean of their respective columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Replace missing categorical values with the most frequent value (mode)
categorical_columns = data.select_dtypes(include=['object']).columns
data[categorical_columns] = data[categorical_columns].apply(lambda col: col.fillna(col.mode()[0]))

print("\nMissing Values After Imputation:")
print(data.isnull().sum())

# Step 2: Encode Categorical Variables
# Convert categorical columns to numeric using one-hot encoding
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Step 3: Scale Numeric Variables (Optional - Normalize data for better model performance)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

print("\nData Preprocessing Complete!")
print("First 5 Rows of Processed Data:")
print(data.head())
