# STEP 1: Import modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# STEP 2: Load the dataset
df = pd.read_csv('deliveries.csv')
df.dropna(inplace=True)

# STEP 3: Feature engineering (adjust as needed based on your dataset)
df['runs_left'] = df['total_runs_x'] - df['score']
df['balls_left'] = 120 - df['overs'] * 6
df['wickets_left'] = 10 - df['wickets']
df['crr'] = df['score'] / df['overs']
df['rrr'] = (df['runs_left'] * 6) / df['balls_left']
df['result'] = df['result'].apply(lambda x: 1 if x == 'win' else 0)  # Adjust if needed

# STEP 4: Select features and target
X = df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left', 'total_runs_x', 'crr', 'rrr']]
y = df['result']

# STEP 5: Preprocessing and model pipeline
categorical_cols = ['batting_team', 'bowling_team', 'city']
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

# STEP 6: Train the model
pipe.fit(X, y)

# STEP 7: Save the pipeline
with open('pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("✅ Model trained and saved as 'pipe.pkl'")
