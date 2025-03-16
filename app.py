import streamlit as st
import pickle
import pandas as pd
import numpy as np
import datetime

# Load the trained model
model_path = r'F:\breast_cancer\KNN.sav'
model = pickle.load(open(model_path, 'rb'))
st.title('ML Model Deployment')

# Input fields for your model (adjust based on your model's expected features)
radius = st.number_input("Mean Radius", value=0.0)
texture = st.number_input("Mean Texture", value=0.0)
perimeter = st.number_input("Mean Perimeter", value=0.0)
area = st.number_input("Mean Area", value=0.0)
smoothness = st.number_input("Mean Smoothness", value=0.0)
compactness = st.number_input("Mean Compactness", value=0.0)

# Collect input into a DataFrame
input_data = pd.DataFrame([[radius, texture, perimeter, area, smoothness, compactness]],
                          columns=['Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness', 'Mean Compactness'])

# Prediction logic
if st.button('Predict'):
    try:
        # If using a scaler, apply it to input data (uncomment if needed)
        # input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)

        # Create report DataFrame
        report_df = input_data.copy()
        report_df['Prediction'] = prediction[0]
        
        # If the model supports probability predictions
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data).max()  # Get the highest probability score
            report_df['Confidence (%)'] = round(probability * 100, 2)
        
        # Convert numerical predictions to labels (if applicable)
        label_mapping = {0: 'Benign', 1: 'Malignant'}  # Adjust according to your model
        report_df['Prediction_Label'] = label_mapping.get(prediction[0], 'Unknown')
        
        # Add timestamp
        report_df['Timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert to CSV
        csv = report_df.to_csv(index=False).encode('utf-8')
        
        # Add download button
        st.download_button("Download Detailed Report", data=csv, file_name="Prediction_Report.csv", mime="text/csv")
        
        # Display explanation on UI
        st.write("### Prediction Explanation")
        st.write(f"🟢 **Prediction:** {report_df['Prediction_Label'][0]}")
        if 'Confidence (%)' in report_df.columns:
            st.write(f"🎯 **Confidence Score:** {report_df['Confidence (%)'][0]}%")
        st.write(f"📅 **Timestamp:** {report_df['Timestamp'][0]}")
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")

  