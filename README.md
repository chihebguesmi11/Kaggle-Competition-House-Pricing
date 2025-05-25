# Kaggle-Competition-House-Pricing
# House Prices Prediction

This repository contains my solution for the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition. The project achieved a rank of **474 out of 4667** participants, placing it in the top 10% on the leaderboard.

## Project Overview

The goal of this project is to predict house sale prices based on various features such as location, size, quality, and other property characteristics. The solution employs advanced feature engineering, multiple machine learning models, and a stacking ensemble to achieve high predictive accuracy.

## Methodology

The code implements a comprehensive data preprocessing and modeling pipeline:

1. **Data Loading and Exploration**:
   - Load training and test datasets (`train.csv`, `test.csv`) using pandas.
   - Perform initial data exploration to understand features and missing values.

2. **Feature Engineering**:
   - **Missing Value Treatment**: Impute missing values for features like `LotFrontage` (neighborhood median), `MasVnrType` (None), and others with appropriate strategies.
   - **Binary Flags**: Create indicators for the presence of features like basements, garages, and fireplaces.
   - **Age-Related Features**: Derive features such as `PropertyAge`, `RecentRemodel`, and `IsNewHome`.
   - **Size and Area Metrics**: Calculate `TotalSF`, `TotalBathrooms`, and ratios like `LivingAreaRatio`.
   - **Quality Encodings**: Map quality-related categorical features to numerical scores (e.g., `ExterQual`, `BsmtQual`).
   - **Compound Metrics**: Create interaction terms like `OverallGrade` and `KitchenScore`.
   - **Neighborhood Analysis**: Derive `MeanNeighborhoodPrice` and `NeighborhoodTier` based on sale price distributions.
   - **Categorical Simplification**: Simplify features like `HouseStyle` and `BldgType`.
   - **Proximity Features**: Add indicators for positive/negative location conditions.
   - **Interaction Terms**: Generate features like `QualitySize` and `AgeQuality`.
   - **Temporal Features**: Create `SeasonSold` and `SaleEra` based on sale dates.
   - **Polynomial and Ratio Features**: Add squared terms and ratios like `RoomDensity`.
   - **Log Transformations**: Apply log transformations to skewed numerical features.
   - **One-Hot Encoding**: Encode categorical variables to avoid multicollinearity.
   - **Feature Selection**: Use RandomForest to select the top 100 features based on importance.

3. **Modeling**:
   - Train multiple models: Ridge, Lasso, RandomForest, XGBoost, and CatBoost.
   - Perform hyperparameter tuning using GridSearchCV with K-fold cross-validation.
   - Implement a stacking ensemble combining the best models, with Ridge as the meta-learner.
   - Use a custom-tuned CatBoost model for final predictions.

4. **Evaluation**:
   - Evaluate models using RMSE, MAE, and R² on a test set.
   - The custom CatBoost model achieved the best performance, used for the final submission.

5. **Submission**:
   - Generate predictions on the test set, inverse-transform log prices, and create `submission9.csv`.

## Results

- **Leaderboard Rank**: 474 / 4667 (Top 10%)
- **Best Model**: Custom-tuned CatBoost Regressor
- **Key Metrics** (on validation set, varies by run):
  - RMSE: ~0.12 (log scale)
  - MAE: ~0.08 (log scale)
  - R²: ~0.90

