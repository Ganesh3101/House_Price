import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df = pd.read_csv("static/data.csv",header=0)
df.dropna(inplace=True,axis=1)
df.drop(columns = ['date','yr_renovated','street','country','statezip','condition','sqft_basement','waterfront','view'],inplace=True)
counts = df['city'].value_counts()
df = df[df['city'].isin(counts[counts >= 10].index)]
df['price'] = df['price'] / 1000
X = df.drop(columns = ['price'])
y = df.price
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=10)
le = LabelEncoder()
X_train.city = le.fit_transform(X_train.city)
X_test.city = le.transform(X_test.city)
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(le,'label_encoder_city.joblib')
joblib.dump(scaler, 'scaled.joblib')

models_and_parameters = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {

        }
    },
    'DecisionTree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [25,50,75,100],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5]
        }
    },

'SVR': {
    'model': SVR(),
    'params': {
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2],
        'kernel': ['rbf']
    }
}

}

best_models = []

for name, mp in models_and_parameters.items():
    print(f"Running GridSearchCV for {name}...")
    grid = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"Best Params: {grid.best_params_}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²: {r2:.4f}\n")

    best_models.append((name, best_model, rmse, r2))

best_models.sort(key=lambda x: x[2])
best_model_name, best_model, best_rmse, best_r2 = best_models[0]

joblib.dump(best_model, f"{best_model_name}_best_model.joblib")
print(f"Saved best model: {best_model_name} with RMSE = {best_rmse:.2f}, R² = {best_r2:.4f}")

print(best_model.score(X_test_scaled,y_test))
