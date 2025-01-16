# Import necessary libraries
import pandas as pd

# Load the dataset
file_path = '/workspaces/Personal-Loan-Default-Prediction/Loan.csv'  # Update with your dataset path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Head:")
print(data.head())

# Display summary information about the dataset
print("\nDataset Info:")
print(data.info())

# Check for missing values in the dataset
print("\nMissing Values:")
print(data.isnull().sum())

