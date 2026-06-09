import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

print("1. Generating HR Data...")
np.random.seed(42)
n_samples = 5000

# Generate 5,000 employees
satisfaction = np.random.randint(1, 11, n_samples)
years = np.random.randint(0, 20, n_samples)
income = np.random.randint(3000, 15000, n_samples)
overtime = np.random.randint(0, 2, n_samples)

# Math to determine who quits
prob = 0.5 - (satisfaction * 0.04) + (overtime * 0.2) - (income * 0.00001) + (np.random.randn(n_samples) * 0.1)
attrition = (prob > 0.45).astype(int)

# Package into a clean DataFrame
X = pd.DataFrame({'Satisfaction': satisfaction, 'Years': years, 'Income': income, 'Overtime': overtime})
y = attrition

print("2. Training Random Forest Algorithm...")
# This single line replaces 30 lines of PyTorch Neural Network code!
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("3. Saving the Brain...")
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("🎉 Complete! Model saved as 'rf_model.pkl'")
