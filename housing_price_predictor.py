# housing_price_predictor.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load

# Load the dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Explore the dataset
def explore_data(data):
    print("First five rows of the dataset:")
    print(data.head())
    print("\nDataset information:")
    print(data.info())
    print("\nDistribution of CHAS feature:")
    print(data["CHAS"].value_counts())
    print("\nStatistical summary of the dataset:")
    print(data.describe())
    data.hist(bins=50, figsize=(20, 15))
    plt.show()

# Split the data into training and test sets
def split_data(data):
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in strat_split.split(data, data["CHAS"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    return strat_train_set, strat_test_set

# Preprocess the data
def preprocess_data(train_set):
    housing = train_set.copy()
    housing["TAXRM"] = housing["TAX"] / housing["RM"]

    corr_matrix = housing.corr()
    print("\nCorrelation matrix:")
    print(corr_matrix["MEDV"].sort_values(ascending=False))

    housing = strat_train_set.drop("MEDV", axis=1)
    housing_labels = strat_train_set["MEDV"].copy()
    return housing, housing_labels

# Build and train the model
def build_and_train_model(housing, housing_labels):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ])
    housing_num_tr = pipeline.fit_transform(housing)

    model = RandomForestRegressor()
    model.fit(housing_num_tr, housing_labels)

    return model, pipeline

# Evaluate the model
def evaluate_model(model, pipeline, strat_train_set, housing_labels):
    some_data = strat_train_set.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    prepared_data = pipeline.transform(some_data)
    predictions = model.predict(prepared_data)
    print("\nSample predictions:")
    print(predictions, list(some_labels))

    housing_num_tr = pipeline.transform(housing)
    housing_predictions = model.predict(housing_num_tr)
    mse = mean_squared_error(housing_labels, housing_predictions)
    rmse = np.sqrt(mse)
    print(f"\nRoot Mean Squared Error (RMSE) on training data: {rmse}")

    scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-scores)
    print("\nCross-validation RMSE scores:")
    print(rmse_scores)
    print(f"Mean RMSE: {rmse_scores.mean()}")
    print(f"Standard Deviation of RMSE: {rmse_scores.std()}")

# Save the model
def save_model(model, filename):
    dump(model, filename)

# Test the model on the test set
def test_model(model, pipeline, test_set):
    X_test = test_set.drop("MEDV", axis=1)
    Y_test = test_set["MEDV"].copy()
    X_test_prepared = pipeline.transform(X_test)
    final_predictions = model.predict(X_test_prepared)
    final_mse = mean_squared_error(Y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(f"\nRoot Mean Squared Error (RMSE) on test data: {final_rmse}")

    print("Test set predictions:")
    print(final_predictions, list(Y_test))

# Load the model and make a prediction
def load_and_predict(filename, features):
    model = load(filename)
    prediction = model.predict(features)
    print("\nPrediction for new data:")
    print(prediction)

# Main function to run the workflow
if __name__ == "__main__":
    # Filepath to the dataset
    filepath = 'UPDATED.csv'

    # Load and explore data
    data = load_data(filepath)
    explore_data(data)

    # Split data
    strat_train_set, strat_test_set = split_data(data)

    # Preprocess data
    housing, housing_labels = preprocess_data(strat_train_set)

    # Build and train model
    model, pipeline = build_and_train_model(housing, housing_labels)

    # Evaluate the model
    evaluate_model(model, pipeline, strat_train_set, housing_labels)

    # Save the model
    save_model(model, 'housing_price_model.joblib')

    # Test the model on the test data
    test_model(model, pipeline, strat_test_set)

    # Example of making a prediction
    features = np.array([[-5.43, 4.1, -1.6, -0.6, -1.4, -11.44, -49.3, 7.6, -26.001, -0.5, -0.97, 0.41164, -66.86]])
    load_and_predict('housing_price_model.joblib', features)
