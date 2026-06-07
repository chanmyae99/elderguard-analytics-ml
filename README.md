# ElderGuard Analytics ML

## Group Information

### Group Name

Team X

### Group Members

| Name               | Contribution                                                                                                                        |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| Chan Myae Aung     | Exploratory Data Analysis (EDA), Project Architecture & Structure Design, Data Service, Evaluation Service, Docker Containerization |
| Sai Thaw Zin Lynn  | Model Development, Training Service, Hyperparameter Tuning, Logistic Regression, Random Forest, XGBoost Models                      |
| See Long Hua Brian | Data Preprocessing Pipeline, Feature Engineering, Class Imbalance Handling (SMOTE)                                                  |

---

# Python Files Written

## Chan Myae Aung

| File                               | Purpose                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------- |
| notebooks/eda.ipynb                | Exploratory Data Analysis                                                       |
| src/services/data_service.py       | Data loading, train-test split and feature preparation                          |
| src/services/evaluation_service.py | Model evaluation, metrics generation, confusion matrices and feature importance |
| Dockerfile                         | Containerization configuration                                                  |
| docker-compose.yml                 | Docker orchestration                                                            |
| run.sh                             | Pipeline startup script                                                         |

## Sai Thaw Zin Lynn

| File                                    | Purpose                            |
| --------------------------------------- | ---------------------------------- |
| src/models/logistic_regression_model.py | Logistic Regression implementation |
| src/models/random_forest_model.py       | Random Forest implementation       |
| src/models/xgboost_model.py             | XGBoost implementation             |
| src/services/training_service.py        | Model training workflow            |
| src/tuning_pipeline.py                  | Hyperparameter tuning workflow     |

## See Long Hua Brian

| File                                   | Purpose                              |
| -------------------------------------- | ------------------------------------ |
| src/preprocessing/data_preprocessor.py | Data cleaning and preprocessing      |
| src/preprocessing/feature_engineer.py  | Feature engineering and encoding     |
| src/preprocessing/imbalance_handler.py | Class imbalance handling using SMOTE |

---

# Project Structure

```text
elderguard-analytics-ml/
│
├── config/
├── data/
│   ├── raw/
│   └── processed/
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── saved_model/
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── services/
│   ├── utils/
│   ├── pipeline.py
│   ├── tuning_pipeline.py
│   └── main.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.sh
```

---

# How to Run the Pipeline

## Step 1: Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / MacOS

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3: Run the Machine Learning Pipeline

```bash
python -m src.main
```

The pipeline performs the following tasks:

1. Data preprocessing
2. Feature engineering
3. Data splitting
4. Model training
5. Model evaluation
6. Best model selection
7. Model persistence

---

# Docker Development Environment

## Build Docker Image

```bash
docker compose build
```

## Run Docker Container

```bash
docker compose up
```

## Stop Docker Container

```bash
docker compose down
```

The Docker container automatically executes the complete machine learning pipeline.

---

# Summary of EDA Findings

The dataset contains environmental sensor readings collected from smart indoor monitoring systems to predict human activity levels.

## Key Findings

* Original dataset contained 10,000 records and 14 columns.
* 171 duplicate records were identified and removed.
* Missing values were handled during preprocessing.
* 795 temperature readings recorded in Kelvin were converted into Celsius.
* 414 records with invalid humidity values were removed.
* Categorical variables contained inconsistent naming and were standardized.
* The target variable (Activity Level) was moderately imbalanced, with the Low Activity class being the majority class.

## Most Important Features

Based on Random Forest feature importance analysis, the most influential features were:

1. MetalOxideSensor_Unit4
2. MetalOxideSensor_Unit2
3. MetalOxideSensor_Unit3
4. CO2_ElectroChemicalSensor
5. MetalOxideSensor_Unit1
6. CO2_InfraredSensor
7. Temperature
8. Humidity
9. CO_GasSensor

These features contributed most significantly to activity level prediction.

---

# Feature Engineering

Several feature engineering techniques were applied to prepare the data for machine learning.

## One-Hot Encoding

Applied to categorical variables:

* HVAC Operation Mode
* Ambient Light Level

This transformed categorical variables into numerical features suitable for machine learning algorithms.

## Label Encoding

Applied to the target variable:

| Activity Level | Encoded Value |
| -------------- | ------------- |
| High           | 0             |
| Low            | 1             |
| Moderate       | 2             |

## Standardization

StandardScaler was applied for Logistic Regression to improve convergence and model performance.

## Feature Removal

Session ID was removed because it acts only as an identifier and does not provide meaningful predictive information.

## Class Imbalance Handling

SMOTE (Synthetic Minority Oversampling Technique) was implemented and evaluated to address class imbalance during experimentation.

---

# Models Used

Three machine learning models were trained and compared.

## Logistic Regression

### Reason for Selection

* Simple and interpretable baseline model
* Fast training and prediction
* Suitable for establishing benchmark performance

---

## Random Forest

### Reason for Selection

* Handles nonlinear relationships effectively
* Robust against noise and outliers
* Provides feature importance analysis
* Reduces overfitting through ensemble learning

---

## XGBoost

### Reason for Selection

* Advanced gradient boosting algorithm
* Captures complex feature interactions
* Generally provides strong predictive performance
* Includes built-in regularization techniques

---

# Hyperparameter Tuning

Hyperparameter tuning was performed using RandomizedSearchCV with 5-fold Cross Validation.

## Random Forest Parameters Tuned

* n_estimators
* max_depth
* min_samples_split
* min_samples_leaf
* max_features
* class_weight

## XGBoost Parameters Tuned

* n_estimators
* learning_rate
* max_depth
* subsample
* colsample_bytree
* reg_alpha
* reg_lambda

The tuning process aimed to maximize model performance while minimizing overfitting.

---

# Evaluation Metrics

Several evaluation metrics were used to assess model performance.

## Accuracy

Measures the proportion of correct predictions made by the model.

## Weighted F1 Score

Weighted F1 Score was selected as the primary evaluation metric because:

* The dataset contains class imbalance.
* It considers both precision and recall.
* It provides a more reliable measure of performance than accuracy alone.

## Macro F1 Score

Macro F1 Score evaluates model performance equally across all classes, regardless of class size.

## Confusion Matrix

Confusion matrices were generated for each model to analyze class-level prediction performance.

---

# Final Model Results

| Model               | Accuracy | Weighted F1 | Macro F1 |
| ------------------- | -------- | ----------- | -------- |
| Logistic Regression | 0.606    | 0.556       | 0.389    |
| Random Forest       | 0.678    | 0.657       | 0.522    |
| XGBoost             | 0.665    | 0.656       | 0.548    |

Random Forest achieved the highest overall performance and was selected as the final model.

---

# Output Generated

The pipeline automatically generates the following outputs:

## Evaluation Reports

```text
reports/metrics/
```

Contains:

* Classification reports
* Model comparison summary
* Feature importance CSV files

## Visualizations

```text
reports/figures/
```

Contains:

* Confusion matrices
* Feature importance plots

## Saved Model

```text
saved_model/best_model.pkl
```

Contains the best-performing trained model selected by the evaluation pipeline.

---

# Conclusion

This project developed a complete end-to-end machine learning pipeline for predicting activity levels from environmental sensor data.

The solution includes:

* Data preprocessing
* Feature engineering
* Class imbalance handling
* Model training
* Hyperparameter tuning
* Model evaluation
* Best model selection
* Model persistence
* Docker containerization

Among the three machine learning models evaluated, Random Forest achieved the best overall performance and was selected as the final model for deployment.
