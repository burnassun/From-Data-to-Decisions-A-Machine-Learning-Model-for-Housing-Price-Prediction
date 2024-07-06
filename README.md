
# Housing Price Predictor

## Overview

This project aims to predict housing prices using machine learning techniques. The goal is to build a model that can estimate property values based on various features such as location, size, and amenities. The project involves data preprocessing, feature engineering, model selection, and evaluation to achieve accurate predictions.

## Features

- **Data Exploration**: Analyzing the dataset to understand the distribution of features and target variables.
- **Feature Engineering**: Creating new features and selecting relevant ones for model building.
- **Model Building**: Comparing different machine learning algorithms to find the best model for price prediction.
- **Model Evaluation**: Using metrics like RMSE and cross-validation to assess the performance of the model.
- **Prediction**: Making price predictions for new housing data.

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

- Python 3.x
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `joblib`

You can install the required packages using `pip`:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

### Data

The dataset used in this project is `UPDATED.csv`, which contains information about housing features and prices. Make sure to place this file in the same directory as the `main.py` script or adjust the file path accordingly.

 
## Code Explanation

- **Data Loading and Exploration**: Load the dataset and perform initial exploration to understand the data.
- **Data Preprocessing**: Handle missing values, encode categorical features, and scale numerical features.
- **Feature Engineering**: Create new features and analyze their impact on the target variable.
- **Model Selection**: Compare different algorithms such as Linear Regression, Decision Tree, and Random Forest.
- **Model Training and Evaluation**: Train the model and evaluate it using metrics like RMSE and cross-validation.
- **Prediction**: Make predictions using the trained model and save it for future use.

## Example

Here’s an example of how to use the trained model to make predictions:

```python
from joblib import load
import numpy as np

# Load the saved model
model = load('housing_price_model.joblib')

# Define the features for prediction
features = np.array([[-5.43, 4.1, -1.6, -0.6, -1.4, -11.44, -49.3, 7.6, -26.001, -0.5, -0.97, 0.41164, -66.86]])

# Make a prediction
prediction = model.predict(features)
print(prediction)
```

## License

This project is licensed under the MIT License. 

