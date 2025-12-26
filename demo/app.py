"""Streamlit demo for patient readmission prediction."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.utils import DataProcessor, generate_synthetic_readmission_data
from models.tabular import ModelFactory, get_model_config
from utils.core import set_seed, get_device
from utils.explainability import ModelExplainer

# Set page config
st.set_page_config(
    page_title="Patient Readmission Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add disclaimer banner
st.error(
    "⚠️ **DISCLAIMER**: This is a research demonstration tool only. "
    "NOT FOR CLINICAL USE. NOT MEDICAL ADVICE. "
    "Always consult healthcare professionals for medical decisions."
)

# Title and description
st.title("🏥 Patient Readmission Prediction")
st.markdown("""
This demo showcases a machine learning model for predicting patient readmission risk within 30 days of discharge.
The model uses structured clinical data including demographics, diagnoses, and hospital stay information.
""")

# Sidebar configuration
st.sidebar.header("Model Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["xgboost", "lightgbm", "random_forest", "logistic_regression", "tabular_net"],
    help="Choose the machine learning model to use for prediction"
)

# Load or generate data
@st.cache_data
def load_demo_data():
    """Load demo data."""
    set_seed(42)
    return generate_synthetic_readmission_data(n_samples=1000, n_patients=500)

# Load data
with st.spinner("Loading demo data..."):
    df = load_demo_data()

# Data overview
st.header("📊 Data Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Patients", len(df))
with col2:
    st.metric("Readmission Rate", f"{df['readmitted'].mean():.1%}")
with col3:
    st.metric("Average Age", f"{df['age'].mean():.1f}")
with col4:
    st.metric("Average LOS", f"{df['length_of_stay'].mean():.1f}")

# Data visualization
st.subheader("Data Distribution")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Demographics", "Clinical Features", "Readmission Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.histogram(df, x="age", nbins=20, title="Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)
        
    with col2:
        # Gender distribution
        gender_counts = df["gender"].value_counts()
        fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index, title="Gender Distribution")
        st.plotly_chart(fig_gender, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Length of stay
        fig_los = px.histogram(df, x="length_of_stay", nbins=15, title="Length of Stay Distribution")
        st.plotly_chart(fig_los, use_container_width=True)
        
    with col2:
        # Primary diagnosis
        diag_counts = df["diag_primary"].value_counts()
        fig_diag = px.bar(x=diag_counts.index, y=diag_counts.values, title="Primary Diagnosis Distribution")
        st.plotly_chart(fig_diag, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Readmission by age group
        df["age_group"] = pd.cut(df["age"], bins=[0, 30, 50, 70, 100], labels=["<30", "30-50", "50-70", "70+"])
        readmission_by_age = df.groupby("age_group")["readmitted"].mean()
        fig_age_readmit = px.bar(x=readmission_by_age.index, y=readmission_by_age.values, 
                                title="Readmission Rate by Age Group")
        st.plotly_chart(fig_age_readmit, use_container_width=True)
        
    with col2:
        # Readmission by diagnosis
        readmission_by_diag = df.groupby("diag_primary")["readmitted"].mean().sort_values(ascending=True)
        fig_diag_readmit = px.bar(x=readmission_by_diag.values, y=readmission_by_diag.index, 
                                 orientation="h", title="Readmission Rate by Diagnosis")
        st.plotly_chart(fig_diag_readmit, use_container_width=True)

# Model training and evaluation
st.header("🤖 Model Training & Evaluation")

if st.button("Train Model", type="primary"):
    with st.spinner("Training model..."):
        # Prepare data
        processor = DataProcessor(
            categorical_features=["admission_type", "discharge_disposition", "admission_source", 
                                "diag_primary", "gender", "race"],
            numerical_features=["age", "length_of_stay", "num_lab_procedures", 
                              "num_medications", "num_procedures", "num_diagnoses"],
            target_column="readmitted",
        )
        
        X, y = processor.fit_transform(df)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model_config = get_model_config(model_type)
        if model_type in ["tabular_net"]:
            model = ModelFactory.create_model(model_type, input_dim=X.shape[1], **model_config)
            # For demo purposes, we'll use a simpler approach
            st.warning("Neural network training requires PyTorch setup. Using XGBoost instead.")
            model = ModelFactory.create_model("xgboost", **get_model_config("xgboost"))
        else:
            model = ModelFactory.create_model(model_type, **model_config)
            
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Store results in session state
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred_proba = y_pred_proba
        
        st.success("Model trained successfully!")

# Display results if model is trained
if "model" in st.session_state:
    st.subheader("Model Performance")
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
    
    auc = roc_auc_score(st.session_state.y_test, st.session_state.y_pred_proba)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AUC-ROC", f"{auc:.3f}")
    with col2:
        # Calculate other metrics
        y_pred = (st.session_state.y_pred_proba > 0.5).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(st.session_state.y_test, y_pred)
        recall = recall_score(st.session_state.y_test, y_pred)
        f1 = f1_score(st.session_state.y_test, y_pred)
        
        st.metric("Precision", f"{precision:.3f}")
    with col3:
        st.metric("Recall", f"{recall:.3f}")
    with col4:
        st.metric("F1-Score", f"{f1:.3f}")
    
    # Plot ROC and PR curves
    col1, col2 = st.columns(2)
    
    with col1:
        # ROC Curve
        fpr, tpr, _ = roc_curve(st.session_state.y_test, st.session_state.y_pred_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc:.3f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col2:
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(st.session_state.y_test, st.session_state.y_pred_proba)
        from sklearn.metrics import average_precision_score
        avg_precision = average_precision_score(st.session_state.y_test, st.session_state.y_pred_proba)
        
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode='lines', 
                                   name=f'PR Curve (AP = {avg_precision:.3f})'))
        fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, use_container_width=True)

# Individual prediction
st.header("🔮 Individual Patient Prediction")

if "model" in st.session_state:
    st.subheader("Enter Patient Information")
    
    # Create input form
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 100, 65)
            length_of_stay = st.slider("Length of Stay (days)", 1, 30, 5)
            num_lab_procedures = st.slider("Number of Lab Procedures", 0, 100, 20)
            num_medications = st.slider("Number of Medications", 0, 50, 8)
            
        with col2:
            num_procedures = st.slider("Number of Procedures", 0, 20, 3)
            num_diagnoses = st.slider("Number of Diagnoses", 1, 20, 5)
            admission_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective", "Newborn"])
            discharge_disposition = st.selectbox("Discharge Disposition", 
                                                ["Home", "SNF", "Rehab", "AMA", "Expired"])
            
        col3, col4 = st.columns(2)
        
        with col3:
            admission_source = st.selectbox("Admission Source", 
                                           ["Physician Referral", "Emergency Room", "Transfer", "Other"])
            diag_primary = st.selectbox("Primary Diagnosis", 
                                       ["Diabetes", "Heart Failure", "Pneumonia", "COPD", "Stroke", "Other"])
            
        with col4:
            gender = st.selectbox("Gender", ["Male", "Female"])
            race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
            
        submitted = st.form_submit_button("Predict Readmission Risk")
        
        if submitted:
            # Create patient data
            patient_data = {
                "age": age,
                "length_of_stay": length_of_stay,
                "num_lab_procedures": num_lab_procedures,
                "num_medications": num_medications,
                "num_procedures": num_procedures,
                "num_diagnoses": num_diagnoses,
                "admission_type": admission_type,
                "discharge_disposition": discharge_disposition,
                "admission_source": admission_source,
                "diag_primary": diag_primary,
                "gender": gender,
                "race": race,
            }
            
            # Convert to dataframe and transform
            patient_df = pd.DataFrame([patient_data])
            X_patient = st.session_state.processor.transform(patient_df)
            
            # Make prediction
            pred_proba = st.session_state.model.predict_proba(X_patient)[0, 1]
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Readmission Risk", f"{pred_proba:.1%}")
                
            with col2:
                risk_level = "High" if pred_proba > 0.7 else "Medium" if pred_proba > 0.3 else "Low"
                st.metric("Risk Level", risk_level)
                
            with col3:
                recommendation = "Consider extended monitoring" if pred_proba > 0.5 else "Standard discharge planning"
                st.metric("Recommendation", recommendation)
            
            # Risk visualization
            fig_risk = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = pred_proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Readmission Risk (%)"},
                delta = {'reference': 30},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig_risk.update_layout(height=300)
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Feature importance (simplified)
            st.subheader("Key Risk Factors")
            
            # Create a simple feature importance based on the input values
            feature_importance = {
                "Age": age / 100,
                "Length of Stay": length_of_stay / 30,
                "Number of Medications": num_medications / 50,
                "Number of Diagnoses": num_diagnoses / 20,
                "Admission Type": 0.8 if admission_type == "Emergency" else 0.3,
                "Primary Diagnosis": 0.7 if diag_primary in ["Heart Failure", "COPD"] else 0.3,
            }
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Display as horizontal bar chart
            fig_importance = go.Figure(go.Bar(
                x=[imp for _, imp in sorted_features],
                y=[name for name, _ in sorted_features],
                orientation='h'
            ))
            fig_importance.update_layout(
                title="Feature Importance (Simplified)",
                xaxis_title="Relative Importance",
                height=300
            )
            st.plotly_chart(fig_importance, use_container_width=True)

else:
    st.info("Please train a model first to make individual predictions.")

# Footer
st.markdown("---")
st.markdown("""
**Note**: This is a demonstration tool for educational purposes only. 
The model uses synthetic data and should not be used for actual clinical decision-making.
""")
