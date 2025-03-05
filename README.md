# Taxi Fare Prediction Challenge

## Overview
This project focuses on building predictive models to estimate the total fare amount paid by travelers for taxi rides. The challenge leverages a comprehensive dataset and applies machine learning techniques to predict the `total_amount` with high accuracy. This project is designed as part of a competition, with performance evaluated using the R² score.

## Dataset
The dataset includes the following files:
- **`train.csv`**: Training set containing the target variable `total_amount` along with other features.
- **`test.csv`**: Test set with similar features but excluding `total_amount`, which must be predicted.
- **`sample_submission.csv`**: A sample submission file provided in the correct format for competition submissions.

### Columns
Key columns in the dataset include:
- **`total_amount`**: The total fare paid for the ride (target variable).
- **`VendorID`**: Identifier for the taxi vendor.
- **`tpep_pickup_datetime`**: Timestamp of the ride pickup.
- **`tpep_dropoff_datetime`**: Timestamp of the ride dropoff.
- **`passenger_count`**: Number of passengers in the ride.
- **`trip_distance`**: Distance covered during the ride (in miles).
- **`RatecodeID`**: Rate code applied to the ride.
- **`store_and_fwd_flag`**: Indicates whether trip data was stored and forwarded.
- **`PULocationID`**: Pickup location ID.
- **`DOLocationID`**: Dropoff location ID.
- **`payment_type`**: Payment method used for the ride.

## Objective
The primary goal is to develop a machine learning model that accurately predicts the `total_amount` based on the provided features. Model performance is evaluated using the **R² score**, a metric that measures the proportion of variance in the target variable explained by the model.

## Approach
The project follows these key steps:
1. **Data Exploration**:
   - Analyze the dataset to understand its structure and distributions.
   - Handle missing values and outliers.
   - Visualize feature relationships and distributions.
2. **Feature Engineering**:
   - Extract meaningful features from `tpep_pickup_datetime` and `tpep_dropoff_datetime` (e.g., trip duration, hour of day).
   - Encode categorical variables (e.g., `VendorID`, `RatecodeID`, `payment_type`).
   - Scale numerical features (e.g., `trip_distance`, `passenger_count`) for better model performance.
3. **Modeling**:
   - Implement and compare multiple machine learning models, including:
     - Decision Trees
     - K-Nearest Neighbors (KNN)
     - Gradient Boosting
4. **Hyperparameter Tuning**:
   - Use grid search and cross-validation to optimize model parameters.
5. **Evaluation**:
   - Assess model performance using the R² score on a validation set.
   - Select the best-performing model for final predictions.

## Results
The project achieved an R² score of approximately **91%** on the test set using an ensemble of models. Continuous iteration and fine-tuning of features and hyperparameters contributed to this performance.

## Installation & Requirements
To run this project, ensure you have Python installed along with the following libraries:
- `numpy`: For numerical computations.
- `pandas`: For data manipulation and analysis.
- `scikit-learn`: For machine learning models and evaluation.
- `matplotlib` (optional): For data visualization.

### Install Dependencies
Install the required packages using pip:
```bash
pip install numpy pandas scikit-learn matplotlib
