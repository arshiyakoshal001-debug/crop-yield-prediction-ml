# Crop Yield Prediction Using Machine Learning
# Self-directed student project using a clearly labeled synthetic dataset.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("agricultural_yield_dataset.csv")

X = df.drop(columns="yield_t_ha")
y = df["yield_t_ha"]

categorical = ["crop"]
numeric = [c for c in X.columns if c not in categorical]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

model = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

# Next step: replace the synthetic dataset with a real, documented agricultural dataset.
