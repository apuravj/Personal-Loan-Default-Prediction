# Import necessary libraries
from sklearn.model_selection import train_test_split

# Step 1: Identify Features and Target
# Replace 'Target_Column' with the name of your target column
if 'Target_Column' not in data.columns:  # Ensure target column exists
    raise ValueError("Please specify the correct target column name!")

X = data.drop(columns=['Target_Column'])  # Features
y = data['Target_Column']  # Target

# Step 2: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Data Splitting Complete:")
print(f"Training Set: {X_train.shape[0]} samples")
print(f"Testing Set: {X_test.shape[0]} samples")

###########################################################################################################

# Import the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Initialize the Model
model = DecisionTreeClassifier(random_state=42)

# Step 2: Train the Model
model.fit(X_train, y_train)

# Step 3: Evaluate the Model
y_pred = model.predict(X_test)

# Step 4: Print Model Performance
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
