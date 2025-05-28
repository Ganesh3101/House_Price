import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

# Load the dataset
df = pd.read_csv("static/AmesHousing.csv",header=0)
df.dropna(inplace=True,axis=1)
print(df.info())

features = [
    'Gr Liv Area', 'Full Bath', 'Bedroom AbvGr', 'TotRms AbvGrd', 'Year Built'
]

target = 'SalePrice'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
models_and_parameters = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {
            # no hyperparameters for GridSearch, included for consistency
        }
    },
    'Ridge': {
        'model': Ridge(),
        'params': {
            'alpha': [0.1, 1.0, 10.0]
        }
    },
    'Lasso': {
        'model': Lasso(),
        'params': {
            'alpha': [0.001, 0.01, 0.1, 1.0]
        }
    },
    'DecisionTree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
    }
}

best_models = []

for name, mp in models_and_parameters.items():
    print(f"Running GridSearchCV for {name}...")
    grid = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Best Params: {grid.best_params_}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²: {r2:.4f}\n")

    best_models.append((name, best_model, rmse, r2))

# Save the best performing model (lowest RMSE)
best_models.sort(key=lambda x: x[2])  # sort by RMSE
best_model_name, best_model, best_rmse, best_r2 = best_models[0]

joblib.dump(best_model, f"{best_model_name}_best_model.joblib")
print(f"Saved best model: {best_model_name} with RMSE = {best_rmse:.2f}, R² = {best_r2:.4f}")

print(best_model.score(X_test,y_test))

# # Save model
# joblib.dump(model, "house_price_model.joblib")
#
# print("Model trained and saved as 'house_price_model.joblib'")
#
# print(model.score(X_test,y_test))
# target = 'SalePrice'
#
# # Drop rows with missing values in selected columns
# df = df.dropna(subset=features + [target])
#
# X = df[features]
# y = df[target]
#
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train Linear Regression
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Predict and evaluate
# y_pred = model.predict(X_test)
# print(f"R^2 Score: {r2_score(y_test, y_pred):.4f}")
# print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
#
# # Save model
# dump(model, 'ames_housing_model.joblib')
# print("Model saved as 'ames_housing_model.joblib'")