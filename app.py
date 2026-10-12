import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# Set the title of the application
st.title("Heart Disease Predictor")

# Create three tabs for different functionalities
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

# ============================================================================
# UTILITY FUNCTION: Create download link for CSV file
# ============================================================================
def get_binary_file_downloader_html(df):
    """
    Generate an HTML link to download a DataFrame as a CSV file.
    
    Args:
        df: Pandas DataFrame to be downloaded
        
    Returns:
        HTML string with download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href

# ============================================================================
# TAB 1: SINGLE PREDICTION
# ============================================================================
with tab1:
    st.subheader("Enter Patient Information")
    
    # Input fields for patient data
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Convert categorical inputs to numeric values for model processing
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'chest_pain': [chest_pain],
        'resting_bp': [resting_bp],
        'cholesterol': [cholesterol],
        'fasting_bs': [fasting_bs],
        'resting_ecg': [resting_ecg],
        'max_hr': [max_hr],
        'exercise_angina': [exercise_angina],
        'oldpeak': [oldpeak],
        'st_slope': [st_slope]
    })

    # Define model names and file paths
    algonames = ['Decision Trees', 'Logistic Regression', 'KNN', 'Support Vector Machine', 'Random Forest (Optimized)']
    modelnames = ['models/DecisionTree_model.pkl', 'models/logistic_regression_model.pkl', 
                  'models/KNN_model.pkl', 'models/SVM_model.pkl', 'models/RandomForest_model.pkl']
    
    def predict_heart_disease(data):
        """
        Make predictions using multiple trained models.
        
        Args:
            data: DataFrame containing patient features
            
        Returns:
            List of predictions from each model
        """
        predictions = []
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions

    # Create a submit button to make predictions
    if st.button("Submit"):
        st.subheader('Results....')
        st.markdown('----------------------------')
        
        # Map lowercase column names to trained model's expected column names
        input_data.rename(columns={
            'age': 'Age',
            'sex': 'Sex',
            'chest_pain': 'ChestPainType',
            'resting_bp': 'RestingBP',
            'cholesterol': 'Cholesterol',
            'fasting_bs': 'FastingBS',
            'resting_ecg': 'RestingECG',
            'max_hr': 'MaxHR',
            'exercise_angina': 'ExerciseAngina',
            'oldpeak': 'Oldpeak',
            'st_slope': 'ST_Slope'
        }, inplace=True)

        # Get predictions from all models
        with st.spinner('🔄 Analyzing patient data with 5 models...'):
            import time
            time.sleep(1)
            result = predict_heart_disease(input_data)
        
        # Display results from each model
        for i in range(len(result)):
            import random
            print(random.randint(1,2000))
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.write("✅ No heart disease detected.")
            else:
                st.write("⚠️ Heart disease detected.")
            st.markdown('-------------------------')

# ============================================================================
# TAB 2: BULK PREDICTION FROM CSV
# ============================================================================
with tab2:
    st.title("Upload CSV File for Bulk Predictions")

    st.subheader('Instructions to note before uploading the file:')
    st.info("""
        1. No NaN values allowed.
        2. Total 11 features in this order: ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
       'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope')
        3. Check the spellings of the feature names. 
        4. Feature values conventions:
            - Age: age of the patient [years]
            - Sex: sex of the patient [0: Male, 1: Female]
            - ChestPainType: chest pain type [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic]
            - RestingBP: resting blood pressure [mm Hg]
            - Cholesterol: serum cholesterol [mm/dl]
            - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
            - RestingECG: resting electrocardiogram results [0: Normal, 1: having ST-T wave abnormality, 2: left ventricular hypertrophy]
            - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
            - ExerciseAngina: exercise-induced angina [1: Yes, 0: No]
            - Oldpeak: oldpeak = ST [Numeric value measured in depression]
            - ST_Slope: the slope of the peak exercise ST segment [0: upsloping, 1: flat, 2: downsloping]
    """)

    # Create a file uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file into a DataFrame
            input_data = pd.read_csv(uploaded_file)
            
            # Load the logistic regression model (corrected filename)
            model = pickle.load(open('models/RandomForest_model.pkl', 'rb'))

            # Define expected columns
            expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                              'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

            # Check if all expected columns are present
            if set(expected_columns).issubset(input_data.columns):
                
                def map_categorical_to_numeric(df):
                    """
                    Convert categorical values to numeric values for model processing.
                    
                    Args:
                        df: DataFrame with categorical values
                        
                    Returns:
                        DataFrame with numeric values
                    """
                    # Define mappings for categorical columns
                    sex_mapping = {'M': 0, 'F': 1}
                    chest_pain_mapping = {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}
                    resting_ecg_mapping = {'Normal': 0, 'ST': 1, 'LVH': 2}
                    exercise_angina_mapping = {'N': 0, 'Y': 1}
                    st_slope_mapping = {'Up': 0, 'Flat': 1, 'Down': 2}

                    # Apply mappings only if column contains categorical values (strings)
                    if df['Sex'].dtype == 'object':
                        df['Sex'] = df['Sex'].map(sex_mapping)
                    if df['ChestPainType'].dtype == 'object':
                        df['ChestPainType'] = df['ChestPainType'].map(chest_pain_mapping)
                    if df['RestingECG'].dtype == 'object':
                        df['RestingECG'] = df['RestingECG'].map(resting_ecg_mapping)
                    if df['ExerciseAngina'].dtype == 'object':
                        df['ExerciseAngina'] = df['ExerciseAngina'].map(exercise_angina_mapping)
                    if df['ST_Slope'].dtype == 'object':
                        df['ST_Slope'] = df['ST_Slope'].map(st_slope_mapping)

                    return df
                
                # Convert categorical values to numeric if present
                input_data = map_categorical_to_numeric(input_data)
                
                # Check for NaN values after mapping
                if input_data[expected_columns].isnull().any().any():
                    st.error("❌ Error: The uploaded file contains NaN values or invalid categorical values. Please check your data.")
                else:
                    # Select only the expected columns in the correct order
                    features_data = input_data[expected_columns]
                    
                    # Make predictions using the model (vectorized operation - much faster)
                    predictions = model.predict(features_data)
                    
                    # Add predictions to the DataFrame using .loc to avoid SettingWithCopyWarning
                    input_data.loc[:, 'Prediction'] = predictions
                    
                    # Map predictions to readable labels
                    input_data.loc[:, 'Prediction_Label'] = input_data['Prediction'].map({
                        0: 'No Heart Disease',
                        1: 'Heart Disease Detected'
                    })
                    
                    # Save to CSV file
                    input_data.to_csv('PredictedHeart.csv', index=False)

                    # Display the predictions
                    st.subheader("✅ Predictions Complete:")
                    st.write(input_data)
                    
                    # Show summary statistics
                    total_patients = len(input_data)
                    disease_count = (input_data['Prediction'] == 1).sum()
                    no_disease_count = (input_data['Prediction'] == 0).sum()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Patients", total_patients)
                    with col2:
                        st.metric("Heart Disease Detected", disease_count)
                    with col3:
                        st.metric("No Heart Disease", no_disease_count)

                    # Create a button to download the updated CSV file
                    st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
            else:
                missing_cols = set(expected_columns) - set(input_data.columns)
                st.warning(f"⚠️ Please make sure the uploaded CSV file has the correct columns. Missing: {missing_cols}")
        
        except FileNotFoundError:
            st.error("❌ Error: Model file not found. Please check the file path.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.info("📤 Upload a CSV file to get predictions.")

# ============================================================================
# TAB 3: MODEL INFORMATION
# ============================================================================
with tab3:
    import plotly.express as px
    
    st.subheader("Model Performance Comparison")
    
    # Model accuracy data
    data = {
        'Decision Trees': 83.69, 
        'Logistic Regression': 82.60, 
        'KNN': 84.78, 
        'Support Vector Machine': 84.23,
        'Random Forest': 86.95
    }
    
    # Create DataFrame for visualization
    Models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=['Models', 'Accuracies'])
    
    # Create interactive bar chart
    fig = px.bar(
        df, 
        y='Accuracies', 
        x='Models',
        title='Model Accuracy Comparison (%)',
        labels={'Accuracies': 'Accuracy (%)', 'Models': 'Model Type'},
        color='Accuracies',
        color_continuous_scale='Blues'
    )
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        height=500
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional model information
    st.markdown("---")
    st.subheader("About the Models")
    st.write("""
    - **Random Forest**: Ensemble method achieving the highest accuracy (86.95%)
    - **KNN**: K-Nearest Neighbors with 84.78% accuracy
    - **Support Vector Machine**: Achieves 84.23% accuracy with optimal hyperparameters
    - **Decision Trees**: Single tree classifier with 83.69% accuracy
    - **Logistic Regression**: Baseline linear model with 82.60% accuracy
    """)