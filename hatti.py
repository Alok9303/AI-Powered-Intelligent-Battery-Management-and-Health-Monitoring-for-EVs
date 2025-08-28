import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from lazypredict.Supervised import  LazyRegressor
import seaborn as sns

# Load dataset
data = pd.read_csv('Battery_RUL.csv')
print(data.columns)

# Detect columns with fewer than 10 unique values
for i in data.columns.values:
    if len(data[i].value_counts().values) < 10:
        print(data[i].value_counts())

# Outlier removal using z-scores
out = []
for i in data.columns.values:
    data['z_scores'] = (data[i] - data[i].mean()) / data[i].std()
    outlier = np.abs(data['z_scores'] > 3).sum()
    if outlier > 3:
        out.append(i)

thresh = 3
for i in out:
    upper = data[i].mean() + thresh * data[i].std()
    lower = data[i].mean() - thresh * data[i].std()
    data = data[(data[i] > lower) & (data[i] < upper)]

# Correlation with RUL
corr = data.corr()['RUL']
corr = corr.drop(['RUL', 'z_scores'])
x_cols = [i for i in corr.index if corr[i] > 0]
x = data[x_cols]
y = data['RUL']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# Initialize models
models = {
    "Extra Trees": ExtraTreesRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

results = {}

# Train models and evaluate
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
    print(f"{name} prediction:", y_pred[:5])

'''# 1. Residual Plot
plt.figure(figsize=(10, 6))
for name in results:
    residuals = y_test - results[name]["y_pred"]
    sns.histplot(residuals, kde=True, label=name, bins=30)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

# 2. Prediction vs Actual
plt.figure(figsize=(10, 6))
for name in results:
    plt.scatter(y_test, results[name]["y_pred"], label=name, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.title("Prediction vs Actual Values")
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.legend()
plt.grid(True)
plt.show()

# 3. Feature Importance (from Extra Trees)
importances = results["Extra Trees"]["model"].feature_importances_
indices = np.argsort(importances)[::-1]
features = x.columns[indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features)
plt.title("Feature Importance - Extra Trees")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.grid(True)
plt.show()

# 4. R² Score Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=[results[m]["r2"] for m in results])
plt.title("R² Score Comparison")
plt.ylabel("R² Score")
plt.ylim(0.95, 1.01)
plt.grid(True, axis='y')
plt.show()

# 5. MSE and RMSE Comparison
mse_vals = [results[m]["mse"] for m in results]
rmse_vals = [results[m]["rmse"] for m in results]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(x=list(results.keys()), y=mse_vals, ax=ax[0])
ax[0].set_title("Mean Squared Error (MSE)")
ax[0].set_ylabel("MSE")
ax[0].grid(True, axis='y')

sns.barplot(x=list(results.keys()), y=rmse_vals, ax=ax[1])
ax[1].set_title("Root Mean Squared Error (RMSE)")
ax[1].set_ylabel("RMSE")
ax[1].grid(True, axis='y')

plt.tight_layout()
plt.show()
'''
#LazyRegressor for benchmarking
lazy = LazyRegressor()
models, predictions = lazy.fit(x_train, x_test, y_train, y_test)
print(models)

'''#LIME explanations
explainer = LimeTabularExplainer(
    training_data=np.array(x_train),
    feature_names=x_cols,
    class_names=['RUL'],
    mode='regression'
)

def save_lime_html(model, model_name, x_test_sample):
    # Create explanation for the first sample in the test set
    exp = explainer.explain_instance(
        data_row=x_test_sample,
        predict_fn=model.predict
    )
    # Save to HTML
    exp.save_to_file(f'{model_name}_lime_explanation.html')
    print(f"LIME explanation saved for {model_name}!")

#Generate LIME explanations for each model
x_test_sample = np.array(x_test.iloc[0])
save_lime_html(ext, "ExtraTrees", x_test_sample)
save_lime_html(rf, "RandomForest", x_test_sample)
save_lime_html(xgb, "XGBoost", x_test_sample)'''