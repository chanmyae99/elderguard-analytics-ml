# ElderGuard Analytics ML
## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EC6B23)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite)
![Git](https://img.shields.io/badge/Git-Version%20Control-F05032?logo=git)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)

## Table of Contents

- [Group Information](#group-information)
- [Python Files Written](#python-files-written)
- [Project Structure](#project-structure)
- [How to Run the Pipeline](#how-to-run-the-pipeline)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Hyperparameter Tuning Pipeline](#hyperparameter-tuning-pipeline)
- [Docker Usage](#docker-usage)
- [Docker Volumes](#docker-volumes)
- [Summary of EDA Findings](#summary-of-eda-findings)
- [Feature Engineering](#feature-engineering)
- [Models Used](#models-used)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Final Model Results](#final-model-results)
- [Output Generated](#output-generated)
- [Conclusion](#conclusion)
- [Future Improvements](#future-improvements)

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
| src/ingestion/sqlite_loader.py     | Data Loader for raw dataset                                                     |
| src/ingestion/csv_loader.py        | Data Loader for processed dataset                                               |
| src/services/data_service.py       | Data loading, train-test split and feature preparation                          |
| src/services/evaluation_service.py | Model evaluation, metrics generation, confusion matrices and feature importance |
| src/pipeline.py                    | Orchestrates the end-to-end machine learning workflow.                          |
| src/main.py                        | Entry point for running the complete machine learning pipeline.                 |
| Dockerfile                         | Containerization configuration                                                  |
| docker-compose.yml                 | Docker orchestration                                                            |
| run.sh                             | Pipeline startup script                                                         |

## Sai Thaw Zin Lynn

| File                                    | Purpose                            |
| --------------------------------------- | ---------------------------------- |
| src/models/base_model.py                | Base Model implementation          |
| src/models/logistic_regression_model.py | Logistic Regression implementation |
| src/models/random_forest_model.py       | Random Forest implementation       |
| src/models/xgboost_model.py             | XGBoost implementation             |
| src/services/training_service.py        | Model training workflow            |
| src/tuning_pipeline.py                  | Hyperparameter tuning workflow     |

## See Long Hua Brian

| File                                   | Purpose                              |
| -------------------------------------- | ------------------------------------ |
| src/preprocessing/data_splitter.py     | Handles train-test splitting         |
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

## How to Run the Pipeline

### 1. Clone the Repository

```bash
git clone https://github.com/chanmyae99/elderguard-analytics-ml.git
cd elderguard-analytics-ml
```

### 2. Create a Python Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Machine Learning Pipeline

```bash
python -m src.main
```


---
# Machine Learning Pipeline

```text
gas_monitoring.db
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
saved_model/best_model.pkl
```

# Hyperparameter Tuning Pipeline

A separate tuning pipeline was developed to support future model improvements without modifying the production pipeline.

The tuning pipeline performs:

- RandomizedSearchCV
- 5-Fold Cross Validation
- Random Forest Hyperparameter Optimization
- XGBoost Hyperparameter Optimization

Pipeline flow:

```text
Training Dataset
        ↓
RandomizedSearchCV
        ↓
Cross Validation
        ↓
Best Parameters
        ↓
Tuning Reports
        ↓
config/config.yaml
```

Outputs will be generated in:

```text
reports/
saved_model/
data/processed/

```
# Docker Usage

Docker is used for both development and deployment.

## Development Environment

A separate tuning pipeline is available for model experimentation and future model improvements.

Run the tuning pipeline:

```bash
docker compose up tuning
```
## Docker Deployment

### Build the Docker Image

```bash
docker compose build
```

### Run the Containerized Pipeline

```bash
docker compose up elderguard-ml
```

The container will automatically:

1. Load the dataset
2. Execute the complete ML pipeline
3. Save trained models
4. Generate evaluation reports

### Stop the Container

```bash
docker compose down
```

---

# Docker Volumes

The following Docker volumes are configured:

```text
./data:/app/data
./reports:/app/reports
./saved_model:/app/saved_model
```

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

# Future Improvements

Although Random Forest achieved the best overall performance, the model showed difficulty distinguishing between Moderate and High activity levels.

Future improvements may include:

- Additional feature engineering
- Collection of more balanced training data
- Ensemble learning techniques
- Automated retraining pipelines
- Continuous model monitoring and performance tracking

The separate hyperparameter tuning pipeline allows future model improvements without modifying the production machine learning pipeline.
