# ml-service/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv('data.csv')

# Define features (X) and target (y)
features = ['base_hp', 'base_0_to_60', 'mod_cost', 'mod_hp_gain']
target = 'predicted_0_to_60'

X = data[features]
y = data[target]

# Split data (optional for this simple case, but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Print model score
print(f"Model R^2 score: {model.score(X_test, y_test)}")

# Save the trained model to a file
joblib.dump(model, 'performance_model.pkl')

print("Model trained and saved as performance_model.pkl")