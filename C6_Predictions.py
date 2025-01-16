# Add predictions to the test set
X_test['Predicted'] = y_pred
X_test['Actual'] = y_test.values

# Save to CSV
X_test.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
