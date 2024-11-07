
# Mental Health Prediction using CatBoost and Optuna

This project aims to predict depression based on various personal and lifestyle features. We use CatBoost as the primary model and employ Optuna for hyperparameter optimization.

## Project Structure

1. **Feature Engineering**: 
   - Handles categorical and numerical features.
   - Applies encoding, feature interactions, and binning.
2. **Optuna for Hyperparameter Tuning**:
   - Utilizes a custom objective function with Optuna to optimize CatBoost hyperparameters.
3. **Cross-Validation**:
   - Employs Stratified K-Fold cross-validation to improve model robustness.
4. **Final Model Training and Prediction**:
   - Uses the optimal parameters from Optuna to train the final model and predict on test data.

## Code Explanation

### Dependencies

```python
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from category_encoders import TargetEncoder
```

### Data Loading and Preprocessing

- **Data Cleaning**: Loads train and test datasets and removes extra spaces in column names.
- **Missing Value Handling**: Fills missing values for categorical columns with "Unknown" and numerical columns with the median.
- **Feature Engineering**:
  - Interaction features (`Age_WorkPressure`, `Academic_WorkPressure`) are created.
  - Target encoding is applied to columns like "City" and "Profession".
  - Age is discretized into bins.

### Model Setup and Training

- **Base Parameters**: The model runs on GPU, with parameters defined for `Logloss` and `AUC`.
- **Hyperparameter Tuning**:
  - An `objective` function defines the parameters to be tuned by Optuna.
  - `StratifiedKFold` is used for cross-validation.
  - The goal is to maximize accuracy across the folds.
- **Final Model**:
  - The model is trained using optimal parameters.
  - Predictions are made on the test data.

### Output

- A CSV file (`submission.csv`) is generated, containing predictions with the column names `id` and `Depression`.

## Running the Code

1. Ensure that `mental/train.csv` and `mental/test.csv` are in the correct directory.
2. Run the script to preprocess data, perform hyperparameter tuning, and make predictions.
3. The prediction file is saved as `submission.csv`.

## Summary

This project demonstrates a robust approach to predicting mental health outcomes, focusing on depression. It uses feature engineering, parameter tuning, and CatBoost’s GPU support to achieve high accuracy and efficiency.
