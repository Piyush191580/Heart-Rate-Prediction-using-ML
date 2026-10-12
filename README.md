# Heart Disease Predictor

This project implements a Heart Disease Predictor using various Machine Learning models and provides a user-friendly interface via a Streamlit web application.

## Project Overview

This project aims to develop a robust Heart Disease Predictor using various machine learning algorithms, culminating in an interactive web application built with Streamlit. The entire process, from raw data to a deployable model, is meticulously documented and implemented.

### 1. Data Preprocessing and Exploratory Data Analysis (EDA)

The foundation of this project is the `heart.csv` dataset, which contains a comprehensive set of clinical parameters for heart disease prediction. The initial steps focus on preparing this data for model training:

*   **Data Loading and Initial Inspection**: The dataset is loaded using the `pandas` library. Initial checks are performed to understand the data's structure, identify data types, review descriptive statistics, and detect any missing values.
*   **Handling Categorical Variables**: Several features in the dataset are categorical (e.g., `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`). These are systematically identified and converted into numerical representations, which is crucial for machine learning algorithms that primarily operate on numerical data.
*   **Imputing Missing Values**: A critical step involves addressing missing data. Specifically, `Cholesterol` values recorded as `0` are identified as erroneous and are replaced with `NaN` (Not a Number). These `NaN` values are then imputed using the K-Nearest Neighbors (KNN) imputer, a method that estimates missing values based on the values of their nearest neighbors, ensuring data quality and preventing skewed model performance.
*   **Feature Engineering**: After imputation, relevant columns are cast to appropriate integer data types for consistency and efficient processing.
*   **Correlation Analysis**: The relationships between all features and, most importantly, between each feature and the target variable (`HeartDisease`), are analyzed using correlation matrices. This helps in understanding which features are most influential in predicting heart disease.
*   **Distribution Analysis**: Extensive visualizations (histograms, count plots, box plots, violin plots, pie charts) are generated using `matplotlib` and `seaborn` to understand the distribution of key features (`Age`, `Sex`, `ChestPainType`, `FastingBS`, `MaxHR`, `Oldpeak`, `ExerciseAngina`, `ST_Slope`) and their individual relationships with the presence of heart disease. This step provides crucial insights into the data's characteristics and potential predictive power.

### 2. Model Selection and Training

A comparative approach is taken by training and evaluating five distinct machine learning models to identify the most effective predictor for heart disease:

*   **Logistic Regression**: This serves as a fundamental linear classification model, providing a baseline for performance comparison. It models the probability of a binary outcome.
*   **Support Vector Machine (SVM)**: A powerful algorithm for classification, SVMs work by finding an optimal hyperplane that best separates data points into different classes. Hyperparameter tuning is performed to select the most suitable kernel (e.g., 'linear', 'rbf') for optimal performance.
*   **Decision Tree Classifier**: A non-parametric supervised learning method used for classification. It works by creating a tree-like model of decisions. Hyperparameters such as `max_depth`, `min_samples_split`, and `min_samples_leaf` are optimized using `GridSearchCV` to prevent overfitting and improve generalization.
*   **K-Nearest Neighbors (KNN)**: An instance-based learning algorithm that classifies a data point based on how its neighbors are classified. Feature scaling (`StandardScaler`) is applied, which is crucial for distance-based algorithms like KNN, and cross-validation is used to determine the optimal number of neighbors (`k`).
*   **Random Forest Classifier**: An ensemble learning method that constructs a multitude of decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. This model is known for its high accuracy and robustness. Hyperparameters are optimized using `RandomizedSearchCV` to find the best combination for improved predictive power.

The dataset is split into an 80% training set and a 20% testing set to ensure unbiased evaluation. After rigorous training and evaluation, the **Random Forest Classifier** consistently demonstrated the highest accuracy (approximately 86.95%) among all models, making it the chosen model for deployment in the web application.

### 3. Code Execution Flow

The project is structured into two main components: a Jupyter Notebook for development and a Streamlit application for deployment.

*   **`heart-disease-prediction.ipynb` (Jupyter Notebook)**: This notebook is the core development environment where the entire machine learning pipeline is executed. It provides a step-by-step walkthrough of:
    *   **Data Loading and Initial Exploration**: Demonstrates how the `heart.csv` dataset is loaded and initially inspected.
    *   **Detailed Data Preprocessing**: Shows the code for handling missing values, converting categorical features, and preparing the data for model training.
    *   **Extensive Exploratory Data Analysis**: Includes various code snippets and visualizations to analyze data distributions and relationships.
    *   **Model Training and Evaluation**: Contains the implementation for training each of the five machine learning models, along with their evaluation metrics (accuracy, f1-score, confusion matrices, MAE, MSE, R2 score).
    *   **Model Comparison**: Visualizes the performance of different models to identify the best one.
    *   **Model Persistence**: The trained models are saved as `.pkl` (pickle) files into the `models/` directory. These serialized models are then loaded by the web application for making predictions without retraining.

*   **`app.py` (Streamlit Web Application)**: This Python script powers the interactive web application, providing a user-friendly interface for heart disease prediction. It leverages the pre-trained models saved from the Jupyter notebook. The application is organized into three distinct tabs:
    *   **Predict Tab**: This tab allows users to input individual patient data through a series of interactive forms (sliders, select boxes, number inputs). Upon submission, the application loads all five pre-trained models and provides real-time predictions from each, indicating whether heart disease is detected or not.
    *   **Bulk Predict Tab**: Designed for efficiency, this tab enables users to upload a CSV file containing multiple patient records. The application processes this file, makes predictions for each patient using the best-performing Random Forest model, and then allows the user to download the updated CSV with the added prediction results. Clear instructions are provided regarding the expected CSV format.
    *   **Model Information Tab**: This tab serves as an informative section, displaying a visual comparison of the accuracy of all trained models using an interactive Plotly bar chart. It also provides brief textual descriptions of each model, offering insights into their methodologies and performance.

### 4. Logic and Model Choices

The project addresses a binary classification problem: predicting the presence (1) or absence (0) of heart disease. The selection of models encompasses a spectrum of algorithms, from simpler linear models (Logistic Regression) to more complex non-linear and ensemble techniques (SVM, Decision Tree, KNN, Random Forest). This diverse approach allows for a comprehensive evaluation of different predictive strategies. The superior performance of the Random Forest Classifier is primarily attributed to its ensemble nature, which combines the predictions of multiple decision trees to reduce variance, improve accuracy, and handle complex, non-linear relationships within the data more effectively than individual models.

## Complete Setup Guide

Follow these steps to set up and run the Heart Disease Predictor project from scratch.

### Prerequisites

*   **Python**: Ensure you have Python 3.8+ installed.
*   **uv**: This project uses `uv` as the package manager. If you don't have `uv` installed, you can install it using pip:
    ```bash
    pip install uv
    ```

### 1. Download and Unzip the Project

Since the project will be provided as a Folder via a Google Drive link, follow these steps:
1.  **Download the Folder** from the provided Google Drive link.
2.  **Unzip the downloaded file** to your desired location.
3.  **Navigate into the project directory** (e.g., `cd "Heart Disease Predictor"`) using your terminal. Ensure you are in the root directory of the project where `app.py` and `requirements.txt` are located.
### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage project dependencies.
```bash
uv venv
```
This command will create a virtual environment named `.venv` in your project directory.

### 3. Activate the Virtual Environment

*   **On Windows:**
    ```bash
    .venv\Scripts\activate
    ```
*   **On macOS/Linux:**
    ```bash
    source .venv/bin/activate
    ```

### 4. Install Dependencies

Install all required libraries using the `requirements.txt` file:
```bash
uv add -r requirements.txt
```
The `requirements.txt` file contains:
```
numpy
pandas
matplotlib
seaborn
scikit-learn
streamlit
base64
plotly
```

### 5. Run the Streamlit Application

Once all dependencies are installed, you can launch the web application:
```bash
streamlit run app.py
```
This command will open the application in your default web browser. If it doesn't open automatically, Streamlit will provide a local URL (e.g., `http://localhost:8501`) that you can navigate to.

### 6. Explore the Jupyter Notebook

To understand the underlying machine learning pipeline, you can open and run the `heart-disease-prediction.ipynb` notebook.
```bash
jupyter notebook heart-disease-prediction.ipynb
```
(You might need to install `jupyter` if you haven't already: `uv add jupyter`)
