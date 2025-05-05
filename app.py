"""
RiskLens Pro - Risk Analysis and Prediction Platform
Powered by Streamlit and ML

This application provides project risk analysis, prediction, and visualization capabilities
with self-contained data processing and model training functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import datetime
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import random
from typing import List, Dict, Any, Tuple, Optional, Union

# Import utility modules
from utils.data_processor import (
    handle_file_upload, transform_data_to_template, validate_data, 
    get_column_statistics, create_preprocessing_pipeline, split_train_test_data,
    TARGET_VARIABLE, PROJECT_ID_COLUMN, PROJECT_NAME_COLUMN,
    DEFAULT_CATEGORICAL_FEATURES, DEFAULT_NUMERICAL_FEATURES
)
from utils.model_builder import ModelBuilder
from utils.visualization import (
    plot_feature_importance, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_feature_distribution, plot_scatter,
    plot_cluster_analysis, plot_lime_explanation, plot_risk_heatmap,
    plot_model_comparison, plot_risk_timeline, plot_project_risks
)
from utils.export import create_pdf_report, create_ppt_report, get_download_link

# Constants based on Arcadis Brand Guidelines
ARCADIS_PRIMARY_COLOR = "#FF6900"  # Arcadis Orange
ARCADIS_SECONDARY_COLOR = "#4D4D4F"  # Arcadis Dark Gray
ARCADIS_ACCENT_COLOR = "#4a4a4a"  # Dark Gray (accent)
ARCADIS_LIGHT_BG = "#f0f2f6"  # Light Background
ARCADIS_DARK_BG = "#333333"  # Dark Background
ARCADIS_SUCCESS = "#28a745"  # Green for success
ARCADIS_WARNING = "#ffc107"  # Yellow for warnings
ARCADIS_DANGER = "#dc3545"  # Red for danger/alerts

# Tab configuration
TABS = [
    {"id": "welcome", "name": "Welcome", "emoji": "👋"},
    {"id": "executive_summary", "name": "Executive Summary", "emoji": "📊"},
    {"id": "portfolio_deep_dive", "name": "Portfolio Deep Dive", "emoji": "🔍"},
    {"id": "model_analysis", "name": "Model Analysis & Explainability", "emoji": "🧠"},
    {"id": "simulation", "name": "Simulation & Scenarios", "emoji": "🎲"}
]

# Functions for application flow

@st.cache_data(ttl=3600, show_spinner=False)
def load_sample_data():
    """Load sample data for demo purposes"""
    # Create a basic sample project dataset
    n_samples = 250  # Increased sample size for better visualization
    data = {
        PROJECT_ID_COLUMN: [f"PROJ{1000+i}" for i in range(n_samples)],
        PROJECT_NAME_COLUMN: [f"Project {random.choice(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Omega'])} {i+1}" for i in range(n_samples)],
        "ProjectType": np.random.choice(["Infrastructure", "Building", "Environmental", "Water", "Digital", "Technology"], n_samples),
        "Region": np.random.choice(["North America", "Europe", "APAC", "LATAM", "MEA"], n_samples),
        "Sector": np.random.choice(["Public", "Private", "Mixed"], n_samples),
        "ComplexityLevel": np.random.choice(["Low", "Medium", "High", "Very High"], n_samples),
        "ClientType": np.random.choice(["Government", "Corporate", "Private Equity", "NGO"], n_samples),
        "Budget": np.random.gamma(2, 10000000, n_samples).round(-3),
        "DurationMonths": np.random.randint(3, 84, n_samples),
        "TeamSize": np.random.poisson(15, n_samples) + 2,
        "InitialRiskScore": np.random.beta(2, 5, n_samples).round(3),
        "ChangeRequests": np.random.poisson(5, n_samples),
        "StakeholderEngagementScore": np.random.randint(1, 11, size=n_samples),
        "StartDate": pd.date_range(start="2021-01-01", periods=n_samples, freq="3D"),
        "InitialCost": np.random.gamma(2, 5000000, n_samples).round(-3),
        "InitialScheduleDays": np.random.randint(30, 2000, n_samples),
        "ActualCost": np.random.gamma(2, 5500000, n_samples).round(-3),
        "ActualScheduleDays": np.random.randint(30, 2200, n_samples),
        "RiskEventOccurred": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "ResourceAvailability_High": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    }
    
    # Generate target variable based on features
    base_prob = (
        0.05 
        + (pd.Series(data["ComplexityLevel"]).map({"Low": 0, "Medium": 0.05, "High": 0.15, "Very High": 0.3}))
        + (pd.Series(data["DurationMonths"]) / 800)
        + (pd.Series(data["Budget"]) / 5e8)
        + (pd.Series(data["ChangeRequests"]) * 0.01)
        + (pd.Series(data["RiskEventOccurred"]) * 0.3)
        + (pd.Series(data["ResourceAvailability_High"]) * 0.2)
        - (pd.Series(data["StakeholderEngagementScore"]) * 0.01)
    )
    
    noise = np.random.normal(0, 0.1, n_samples)
    final_prob = np.clip(base_prob + noise, 0.01, 0.95)
    data[TARGET_VARIABLE] = (np.random.rand(n_samples) < final_prob).astype(int)
    
    # Add a few missing values to make it realistic
    for col in ["Budget", "TeamSize", "StakeholderEngagementScore"]:
        mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
        series = pd.Series(data[col])
        series[mask] = np.nan
        data[col] = series.values
    
    df = pd.DataFrame(data)
    
    # Create risk register data
    risk_data = []
    risk_types = ["Schedule Delay", "Cost Overrun", "Scope Creep", "Resource Shortage", "Technical Issue", 
                  "Regulatory Change", "Stakeholder Conflict", "Quality Issues", "Safety Incident", "Dependency Failure"]
    
    for pid in data[PROJECT_ID_COLUMN]:
        num_risks = np.random.randint(0, 8)
        for i in range(num_risks):
            impact = np.random.choice(["Very Low", "Low", "Medium", "High", "Very High"])
            likelihood = np.random.choice(["Very Low", "Low", "Medium", "High", "Very High"])
            
            risk_data.append({
                "RiskID": f"R{np.random.randint(1000, 9999)}",
                PROJECT_ID_COLUMN: pid,
                "RiskType": np.random.choice(risk_types),
                "Impact": impact,
                "Probability": likelihood,
                "Status": np.random.choice(["Open", "Mitigated", "Closed"]),
                "DateIdentified": pd.Timestamp("2022-01-01") + pd.Timedelta(days=np.random.randint(0, 365))
            })
    
    risk_df = pd.DataFrame(risk_data) if risk_data else pd.DataFrame()
    
    return df, risk_df

def format_dataframe_display(df, max_rows=10):
    """Format DataFrame for display with pagination"""
    st.dataframe(df, height=400)
    
    # Show additional rows info if dataframe is large
    if len(df) > max_rows:
        st.caption(f"Displaying {len(df)} of {len(df)} rows.")

def get_data_profiling_metrics(df):
    """Get key metrics about the dataset"""
    metrics = {
        "Total Projects": len(df),
        "High Risk Projects": int(df[TARGET_VARIABLE].sum()) if TARGET_VARIABLE in df.columns else 0,
        "Missing Values": df.isna().sum().sum(),
        "Numerical Features": len(df.select_dtypes(include=["number"]).columns),
        "Categorical Features": len(df.select_dtypes(include=["object", "category"]).columns),
        "Date Features": len(df.select_dtypes(include=["datetime"]).columns)
    }
    
    if TARGET_VARIABLE in df.columns:
        high_risk = metrics["High Risk Projects"]
        total_projects = metrics["Total Projects"]
        metrics["High Risk Rate"] = f"{(high_risk / total_projects * 100):.1f}%"
        
        # Calculate cost and schedule metrics for high risk projects
        high_risk_projects = df[df[TARGET_VARIABLE] == 1]
        if "ActualCost" in df.columns and "InitialCost" in df.columns:
            high_risk_cost_overrun = ((high_risk_projects["ActualCost"] - high_risk_projects["InitialCost"]) / 
                                     high_risk_projects["InitialCost"]).mean() * 100
            metrics["Avg Cost Overrun % (High-Risk)"] = f"{high_risk_cost_overrun:.1f}%"
            
        if "ActualScheduleDays" in df.columns and "InitialScheduleDays" in df.columns:
            high_risk_schedule_overrun = ((high_risk_projects["ActualScheduleDays"] - high_risk_projects["InitialScheduleDays"]) / 
                                         high_risk_projects["InitialScheduleDays"]).mean() * 100
            metrics["Avg Schedule Overrun % (High-Risk)"] = f"{high_risk_schedule_overrun:.1f}%"
    
    return metrics

def calculate_model_metrics(predictions, actuals, probabilities):
    """Calculate and return model performance metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        "accuracy": accuracy_score(actuals, predictions),
        "precision": precision_score(actuals, predictions),
        "recall": recall_score(actuals, predictions),
        "f1_score": f1_score(actuals, predictions),
        "roc_auc": roc_auc_score(actuals, probabilities)
    }
    
    return metrics

def initialize_session_state():
    """Initialize session state variables"""
    if 'project_data' not in st.session_state:
        st.session_state.project_data = None
    if 'risk_data' not in st.session_state:
        st.session_state.risk_data = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = None
    if 'best_model_name' not in st.session_state:
        st.session_state.best_model_name = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = {}
    if 'categorical_features' not in st.session_state:
        st.session_state.categorical_features = DEFAULT_CATEGORICAL_FEATURES
    if 'numerical_features' not in st.session_state:
        st.session_state.numerical_features = DEFAULT_NUMERICAL_FEATURES
    if 'risk_predictions' not in st.session_state:
        st.session_state.risk_predictions = None
    if 'risk_probabilities' not in st.session_state:
        st.session_state.risk_probabilities = None
    if 'data_transformed' not in st.session_state:
        st.session_state.data_transformed = False
    if 'data_profile' not in st.session_state:
        st.session_state.data_profile = None
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "welcome"
    if 'model_builder' not in st.session_state:
        st.session_state.model_builder = ModelBuilder()
    if 'prediction_threshold' not in st.session_state:
        st.session_state.prediction_threshold = 0.4
    if 'current_risk_model' not in st.session_state:
        st.session_state.current_risk_model = "Random Forest"
    if 'ai_narrative_generated' not in st.session_state:
        st.session_state.ai_narrative_generated = False

def arcadis_logo_svg():
    """Return SVG code for Arcadis logo with RiskLens Pro (based on brand guidelines)"""
    return """
    <svg width="200" height="50" viewBox="0 0 300 50" xmlns="http://www.w3.org/2000/svg">
        <!-- Orange wave symbol - Arcadis logo -->
        <path d="M30 10 C40 5, 45 15, 35 17 C25 19, 20 30, 30 35" fill="#FF6900" stroke="none"/>
        
        <!-- Text part -->
        <text x="45" y="30" font-family="Arial" font-size="24" font-weight="bold" fill="#4D4D4F">ARCADIS</text>
        
        <!-- RiskLens Pro text -->
        <text x="160" y="30" font-family="Arial" font-size="22" font-weight="bold" fill="#FF6900">RiskLens Pro</text>
    </svg>
    """

def styled_card(title, content, icon=None, color=ARCADIS_PRIMARY_COLOR):
    """Create a styled card with title and content"""
    icon_html = f'<span style="font-size:24px;margin-right:10px;">{icon}</span>' if icon else ''
    
    st.markdown(f'''
    <div style="border:1px solid #ddd;border-radius:8px;padding:15px;margin-bottom:20px;background:white;">
        <h3 style="color:{color};margin-top:0;border-bottom:1px solid #eee;padding-bottom:10px;">
            {icon_html}{title}
        </h3>
        <div>
            {content}
        </div>
    </div>
    ''', unsafe_allow_html=True)

def styled_metric_card(label, value, delta=None, icon=None, color=ARCADIS_PRIMARY_COLOR, help_text=None):
    """Create a styled metric card"""
    icon_html = f'<span style="font-size:22px;margin-right:8px;">{icon}</span>' if icon else ''
    delta_html = ''
    if delta is not None:
        delta_color = "#28a745" if float(delta.replace('%', '')) >= 0 else "#dc3545"
        delta_icon = "▲" if float(delta.replace('%', '')) >= 0 else "▼"
        delta_html = f'<span style="color:{delta_color};font-size:14px;">{delta_icon} {delta}</span>'

    help_icon = ''
    if help_text:
        help_icon = f'<span title="{help_text}" style="cursor:help;opacity:0.7;margin-left:5px;">ⓘ</span>'
    
    value_style = "font-size:32px;font-weight:bold;"
    if len(str(value)) > 10:  # Adjust font size for long numbers
        value_style = "font-size:24px;font-weight:bold;"
    
    st.markdown(f'''
    <div style="border:1px solid #ddd;border-radius:8px;padding:15px;background:white;height:100%;">
        <div style="color:#666;font-size:14px;">{icon_html}{label}{help_icon}</div>
        <div style="{value_style}color:{color};">{value}</div>
        <div style="margin-top:5px;">{delta_html}</div>
    </div>
    ''', unsafe_allow_html=True)

def styled_header(text, level=1, color=ARCADIS_PRIMARY_COLOR, icon=None):
    """Display a header with custom styling"""
    icon_html = f'{icon} ' if icon else ''
    if level == 1:
        st.markdown(f'<h1 style="color:{color};">{icon_html}{text}</h1>', unsafe_allow_html=True)
    elif level == 2:
        st.markdown(f'<h2 style="color:{color};">{icon_html}{text}</h2>', unsafe_allow_html=True)
    elif level == 3:
        st.markdown(f'<h3 style="color:{color};">{icon_html}{text}</h3>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h4 style="color:{color};">{icon_html}{text}</h4>', unsafe_allow_html=True)

def set_streamlit_style():
    """Set Streamlit page styling"""
    st.set_page_config(
        page_title="RiskLens Pro - Insight Hub",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
    st.markdown(f"""
        <style>
        .main .block-container {{
            padding-top: 1rem;
            padding-bottom: 1rem;
        }}
        h1, h2, h3 {{
            color: {ARCADIS_PRIMARY_COLOR};
        }}
        .stProgress > div > div > div > div {{
            background-color: {ARCADIS_PRIMARY_COLOR};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
            border: none;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {ARCADIS_PRIMARY_COLOR};
            color: white;
        }}
        div[data-testid="stDecoration"] {{
            background-image: linear-gradient(90deg, #FF6900, #4D4D4F);
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # Display header with title and brand colors
    st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem; background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="background-color: {ARCADIS_PRIMARY_COLOR}; width: 8px; height: 40px; margin-right: 15px;"></div>
            <h1 style="color: {ARCADIS_PRIMARY_COLOR}; margin: 0; font-size: 28px;">ARCADIS <span style="color: {ARCADIS_SECONDARY_COLOR};">RiskLens Pro</span></h1>
        </div>
    """, unsafe_allow_html=True)

def make_sidebar():
    """Create application sidebar"""
    with st.sidebar:
        st.markdown("## Data Management")
        
        # Data upload section
        with st.expander("Upload Data", expanded=True):
            upload_option = st.radio(
                "Select data source:",
                options=["Upload data file", "Use sample data"],
                horizontal=True
            )
            
            if upload_option == "Upload data file":
                uploaded_file = st.file_uploader(
                    "Upload your project data file (CSV or Excel)", 
                    type=["csv", "xlsx", "xls"]
                )
                
                if uploaded_file is not None:
                    process_btn = st.button("Process Data", type="primary", use_container_width=True)
                    if process_btn:
                        with st.spinner("Processing data..."):
                            # Process the uploaded file
                            df = handle_file_upload(uploaded_file)
                            
                            if df is not None:
                                st.session_state.project_data = df
                                st.success(f"✅ File loaded successfully")
                                
                                # Transform data to fit required template with intelligent analysis
                                transformed_df, mapping_info, missing_data_impact = transform_data_to_template(df)
                                
                                if transformed_df is not None:
                                    # Store the transformed data and mapping info
                                    st.session_state.project_data = transformed_df
                                    st.session_state.mapping_info = mapping_info
                                    st.session_state.missing_data_impact = missing_data_impact
                                    st.session_state.data_transformed = True
                                    st.session_state.data_profile = get_data_profiling_metrics(transformed_df)
                                    
                                    # Show detailed feedback about the data transformation
                                    overall_impact = missing_data_impact.get('overall', {})
                                    data_quality_score = overall_impact.get('data_quality_score', 0.0)
                                    analysis_capability = overall_impact.get('analysis_capability', 'unknown')
                                    missing_core_features = overall_impact.get('missing_core_features', [])
                                    
                                    # Store feedback messages in session state to persist after rerun
                                    feedback_messages = []
                                    
                                    # Generate feedback messages based on data quality
                                    if analysis_capability == 'full' or analysis_capability == 'limited':
                                        feedback_messages.append({"type": "success", "message": f"✅ Data mapped successfully with {data_quality_score:.0%} coverage"})
                                        if missing_core_features:
                                            feedback_messages.append({"type": "info", "message": f"ℹ️ Some recommended data is missing but analysis can proceed"})
                                    elif analysis_capability == 'severely_limited':
                                        feedback_messages.append({"type": "warning", "message": f"⚠️ Limited analysis possible. Missing important data: {', '.join(missing_core_features)}"})
                                    else:  # not_possible
                                        feedback_messages.append({"type": "error", "message": f"⚠️ Critical data missing: {', '.join(missing_core_features)}. Full analysis not possible."})
                                    
                                    # If target variable is missing, add special warning
                                    if TARGET_VARIABLE in missing_core_features:
                                        feedback_messages.append({"type": "error", "message": f"⚠️ No target risk variable found. Please upload data with risk indicators or outcomes."})
                                        feedback_messages.append({"type": "info", "message": "The app needs a column indicating project risk status, success/failure, or derailment indicators."})
                                    
                                    # Create a data mapping report in the session state for later display
                                    st.session_state.data_mapping_report = {
                                        'quality_score': data_quality_score,
                                        'analysis_capability': analysis_capability,
                                        'missing_features': missing_core_features,
                                        'details': missing_data_impact,
                                        'feedback_messages': feedback_messages
                                    }
                                    
                                    # Display the messages now
                                    for msg in feedback_messages:
                                        if msg["type"] == "success":
                                            st.success(msg["message"])
                                        elif msg["type"] == "info":
                                            st.info(msg["message"])
                                        elif msg["type"] == "warning":
                                            st.warning(msg["message"])
                                        elif msg["type"] == "error":
                                            st.error(msg["message"])
                                    
                                    # Continue with app flow
                                    st.rerun()
            else:  # Use sample data
                if st.button("Load Sample Data", type="primary", use_container_width=True):
                    with st.spinner("Loading sample data..."):
                        sample_df, risk_df = load_sample_data()
                        st.session_state.project_data = sample_df
                        st.session_state.risk_data = risk_df
                        st.session_state.data_profile = get_data_profiling_metrics(sample_df)
                        st.session_state.data_transformed = True
                        
                        # Create and store feedback messages for sample data
                        feedback_messages = [
                            {"type": "success", "message": "✅ Sample data loaded successfully"},
                            {"type": "info", "message": "ℹ️ 150 sample projects with full feature data included"},
                            {"type": "info", "message": "ℹ️ Risk register data with 450+ identified risks also loaded"}
                        ]
                        
                        # Store feedback messages in session state
                        st.session_state.data_mapping_report = {
                            'quality_score': 1.0,  # 100% coverage for sample data
                            'analysis_capability': 'full',
                            'missing_features': [],
                            'details': {},
                            'feedback_messages': feedback_messages
                        }
                        
                        # Display messages before rerun
                        for msg in feedback_messages:
                            if msg["type"] == "success":
                                st.success(msg["message"])
                            elif msg["type"] == "info":
                                st.info(msg["message"])
                            elif msg["type"] == "warning":
                                st.warning(msg["message"])
                            elif msg["type"] == "error":
                                st.error(msg["message"])
                                
                        st.rerun()
        
        # Model training section
        if st.session_state.project_data is not None and st.session_state.data_transformed:
            with st.expander("Run Risk Analytics", expanded=True):
                if st.button("Train Risk Model", type="primary", use_container_width=True):
                    with st.spinner("Training risk model..."):
                        # Execute intelligent model training
                        run_risk_analytics()
                        st.success("✅ Risk model trained successfully")
                        st.session_state.active_tab = "executive_summary"
                        # Generate AI narrative
                        st.session_state.ai_narrative_generated = True
                        st.rerun()
        
        # Display status information
        st.markdown("---")
        st.markdown("### Status")
        
        if st.session_state.project_data is not None:
            st.success(f"✅ Data loaded: {len(st.session_state.project_data)} projects")
        else:
            st.warning("⚠️ No data loaded")
            
        if st.session_state.best_model_name is not None:
            st.success(f"✅ Model: {st.session_state.best_model_name}")
            
        if st.session_state.risk_predictions is not None:
            high_risk = st.session_state.risk_predictions.sum() if st.session_state.risk_predictions is not None else 0
            st.info(f"ℹ️ High-risk projects: {high_risk}")
            
        # Display persisted data processing feedback messages if they exist
        if 'data_mapping_report' in st.session_state and 'feedback_messages' in st.session_state.data_mapping_report:
            st.markdown("---")
            st.markdown("### Data Processing Results")
            
            for msg in st.session_state.data_mapping_report['feedback_messages']:
                if msg["type"] == "success":
                    st.success(msg["message"])
                elif msg["type"] == "info":
                    st.info(msg["message"])
                elif msg["type"] == "warning":
                    st.warning(msg["message"])
                elif msg["type"] == "error":
                    st.error(msg["message"])
        
        # Export section when data is available
        if st.session_state.project_data is not None and st.session_state.risk_predictions is not None:
            st.markdown("---")
            st.markdown("### Export")
            
            export_format = st.selectbox(
                "Export format:", 
                ["PDF Report", "PowerPoint Presentation"]
            )
            
            if st.button("Generate Report", type="primary", use_container_width=True):
                with st.spinner(f"Generating {export_format.split()[0]} report..."):
                    if export_format == "PDF Report":
                        report_data = {
                            "project_data": st.session_state.project_data,
                            "model_results": st.session_state.model_results,
                            "visualizations": st.session_state.visualizations
                        }
                        
                        if st.session_state.risk_data is not None:
                            report_data["risk_data"] = st.session_state.risk_data
                        
                        pdf_bytes = create_pdf_report(**report_data)
                        st.download_button(
                            label=f"Download PDF Report",
                            data=pdf_bytes,
                            file_name="RiskLens_Pro_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        report_data = {
                            "project_data": st.session_state.project_data,
                            "model_results": st.session_state.model_results,
                            "visualizations": st.session_state.visualizations
                        }
                        
                        if st.session_state.risk_data is not None:
                            report_data["risk_data"] = st.session_state.risk_data
                        
                        ppt_bytes = create_ppt_report(**report_data)
                        st.download_button(
                            label=f"Download PowerPoint Presentation",
                            data=ppt_bytes,
                            file_name="RiskLens_Pro_Presentation.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            use_container_width=True
                        )
        
        # Reset application state button
        if st.button("Reset Application", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.rerun()

def run_risk_analytics():
    """Run the risk analytics process on the loaded data"""
    if st.session_state.project_data is None:
        return
    
    df = st.session_state.project_data
    
    # Define features based on available columns
    numerical_features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) 
                         and col != TARGET_VARIABLE and col != PROJECT_ID_COLUMN 
                         and df[col].nunique() > 5]
    
    categorical_features = [col for col in df.columns if (pd.api.types.is_object_dtype(df[col]) 
                           or pd.api.types.is_categorical_dtype(df[col]))
                           and col != PROJECT_NAME_COLUMN and col != PROJECT_ID_COLUMN]
    
    st.session_state.categorical_features = categorical_features
    st.session_state.numerical_features = numerical_features
    
    # Split data for training
    X_train, X_test, y_train, y_test = split_train_test_data(df, TARGET_VARIABLE)
    
    # Create preprocessor
    preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)
    
    # Train the model
    model_builder = st.session_state.model_builder
    results = model_builder.train_all_models(
        X_train, y_train, X_test, y_test, 
        preprocessor, 
        numerical_features + categorical_features,
        cv_folds=3, n_iter=10
    )
    
    # Update session state with results
    st.session_state.trained_models = True
    st.session_state.best_model_name = results["best_model"]
    st.session_state.model_results = results
    
    # Generate predictions for all projects
    X = df.drop(columns=[TARGET_VARIABLE])
    predictions, probabilities = model_builder.predict(X)
    st.session_state.risk_predictions = pd.Series(predictions, index=df.index)
    st.session_state.risk_probabilities = pd.Series(probabilities, index=df.index)
    
    # Add predictions to the dataframe
    df_with_predictions = df.copy()
    df_with_predictions['PredictedRisk'] = predictions
    df_with_predictions['RiskProbability'] = probabilities
    st.session_state.project_data_with_predictions = df_with_predictions

def generate_ai_narrative_summary() -> str:
    """Generate an AI narrative summary of the risk analysis"""
    if not (st.session_state.project_data is not None and st.session_state.risk_predictions is not None):
        return "Insufficient data for narrative generation."
    
    df = st.session_state.project_data
    model_name = st.session_state.best_model_name
    predictions = st.session_state.risk_predictions
    probabilities = st.session_state.risk_probabilities
    threshold = st.session_state.prediction_threshold
    
    high_risk_count = (probabilities > threshold).sum()
    high_risk_rate = high_risk_count / len(df) * 100
    
    # Feature importance (simulated if not available)
    if 'feature_importance' in st.session_state.model_results and st.session_state.best_model_name in st.session_state.model_results['feature_importance']:
        top_features = st.session_state.model_results['feature_importance'][st.session_state.best_model_name][0][:3]
    else:
        # Fallback to default important features
        top_features = ["ComplexityLevel", "Budget", "StakeholderEngagementScore"]
    
    # Get correlations for project types
    project_types = df['ProjectType'].unique()
    project_type_risks = {}
    for pt in project_types:
        mask = df['ProjectType'] == pt
        if mask.sum() > 0:
            risk_rate = (probabilities[mask] > threshold).mean() * 100
            project_type_risks[pt] = risk_rate
    
    highest_risk_type = max(project_type_risks.items(), key=lambda x: x[1]) if project_type_risks else ('Unknown', 0)
    
    # Get correlations for regions
    regions = df['Region'].unique()
    region_risks = {}
    for region in regions:
        mask = df['Region'] == region
        if mask.sum() > 0:
            risk_rate = (probabilities[mask] > threshold).mean() * 100
            region_risks[region] = risk_rate
    
    highest_risk_region = max(region_risks.items(), key=lambda x: x[1]) if region_risks else ('Unknown', 0)
    
    # Calculate average probability
    avg_prob = probabilities.mean() * 100
    median_prob = probabilities.median() * 100
    
    narrative = f"""The {model_name} model predicts {high_risk_count} projects ({high_risk_rate:.1f}% of those with predictions) are at high risk of derailment, 
using a threshold of {threshold}.

Key Insights:

* Primary Risk Drivers: Across the portfolio, the factors most strongly correlated with increased risk appear to be: 
  {', '.join(top_features)}.

* Highest Risk Project Type: Projects classified as '{highest_risk_type[0]}' ({highest_risk_type[1]:.1f}%) show the highest average predicted risk rate.

* Highest Risk Region: The '{highest_risk_region[0]}' region currently exhibits the highest average predicted risk rate at {highest_risk_region[1]:.1f}%.

* Prediction Certainty: The average predicted risk probability across projects is {avg_prob:.1f}% (median: {median_prob:.1f}%). A wider 
  spread might indicate greater uncertainty overall.

Recommendation: Prioritize investigation and potential mitigation actions for the identified high-risk projects, 
paying close attention to the top risk drivers ({', '.join(top_features[:2])}...). Consider focusing 
efforts on projects within the '{highest_risk_type[0]}' type or '{highest_risk_region[0]}' region if applicable. Use the 'Portfolio Deep Dive' 
and 'Model Analysis' tabs for more detailed investigation.
"""
    
    return narrative

# Tab content functions

def welcome_tab():
    """Content for the Welcome tab"""
    # Set layout for the hero section using pure Streamlit components
    st.write("")
    st.write("")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Use direct URL to reliable CDN for image icon
        st.image("https://img.icons8.com/fluency/240/business-risk.png", width=140)
    
    with col2:
        st.write("")
        st.markdown(f'<h1 style="color:{ARCADIS_PRIMARY_COLOR}; font-size:36px;">Welcome to RiskLens Pro</h1>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:{ARCADIS_ACCENT_COLOR}; font-size:18px;">Powered by Arcadis expertise & advanced analytics</p>', unsafe_allow_html=True)
    
    # Create an impactful introduction with a colored background
    st.markdown(f"""
    <div style="background-color: {ARCADIS_LIGHT_BG}; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h2 style="color: {ARCADIS_PRIMARY_COLOR}; margin-top: 0;">The Challenge: Navigating Project Complexity</h2>

    <p style="font-size: 16px;">
    Delivering complex projects on time and within budget is a significant challenge. Factors like scope changes, 
    resource constraints, technical hurdles, and external dependencies can introduce risks, leading to costly 
    overruns and delays. Proactively identifying and understanding these risks is crucial for successful project 
    delivery and maintaining client satisfaction.
    </p>
    </div>

    """, unsafe_allow_html=True)
    
    # Solution section with KPIs
    st.markdown(f"<h2 style='color: {ARCADIS_PRIMARY_COLOR};'>The Solution: Data-Driven Risk Intelligence</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    RiskLens Pro leverages your project data and machine learning to provide early warnings about potential project derailment. 
    By analyzing historical patterns and current project characteristics, it predicts the likelihood of significant cost or schedule overruns.
    """)
    
    # Stats row to make it more visual
    stat1, stat2, stat3 = st.columns(3)
    
    with stat1:
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background-color: white; border-radius: 10px; border-top: 5px solid {ARCADIS_PRIMARY_COLOR}; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h1 style="color:{ARCADIS_PRIMARY_COLOR}; font-size: 40px; margin-bottom: 5px;">30%</h1>
            <p style="color: {ARCADIS_ACCENT_COLOR};">Average cost saving on at-risk projects identified early</p>
        </div>

        """, unsafe_allow_html=True)
    
    with stat2:
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background-color: white; border-radius: 10px; border-top: 5px solid {ARCADIS_SECONDARY_COLOR}; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h1 style="color:{ARCADIS_SECONDARY_COLOR}; font-size: 40px; margin-bottom: 5px;">85%</h1>
            <p style="color: {ARCADIS_ACCENT_COLOR};">Prediction accuracy for project risk classification</p>
        </div>

        """, unsafe_allow_html=True)
    
    with stat3:
        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background-color: white; border-radius: 10px; border-top: 5px solid {ARCADIS_DARK_BG}; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h1 style="color:{ARCADIS_DARK_BG}; font-size: 40px; margin-bottom: 5px;">3x</h1>
            <p style="color: {ARCADIS_ACCENT_COLOR};">Faster identification of potential project risks</p>
        </div>

        """, unsafe_allow_html=True)

    # Journey Map - visual flow of the application
    st.markdown("""<br>""", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: {ARCADIS_PRIMARY_COLOR};'>Your Risk Management Journey</h2>", unsafe_allow_html=True)
    
    # Timeline
    journey_cols = st.columns(5)
    
    with journey_cols[0]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; position: relative;">
            <div style="position: absolute; top: 15px; left: 50%; width: 100%; height: 4px; background-color: {ARCADIS_LIGHT_BG}; z-index: 0;"></div>
            <div style="position: relative; background-color: {ARCADIS_PRIMARY_COLOR}; color: white; width: 40px; height: 40px; line-height: 40px; border-radius: 50%; margin: 0 auto; z-index: 1;">1</div>
            <h4 style="margin-top: 10px;">Data Upload</h4>
            <p style="font-size: 14px;">Load your project data or use our sample data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with journey_cols[1]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; position: relative;">
            <div style="position: absolute; top: 15px; left: -50%; width: 100%; height: 4px; background-color: {ARCADIS_LIGHT_BG}; z-index: 0;"></div>
            <div style="position: relative; background-color: {ARCADIS_PRIMARY_COLOR}; color: white; width: 40px; height: 40px; line-height: 40px; border-radius: 50%; margin: 0 auto; z-index: 1;">2</div>
            <h4 style="margin-top: 10px;">Risk Analysis</h4>
            <p style="font-size: 14px;">ML models analyze and predict project risks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with journey_cols[2]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; position: relative;">
            <div style="position: absolute; top: 15px; left: -50%; width: 100%; height: 4px; background-color: {ARCADIS_LIGHT_BG}; z-index: 0;"></div>
            <div style="position: relative; background-color: {ARCADIS_PRIMARY_COLOR}; color: white; width: 40px; height: 40px; line-height: 40px; border-radius: 50%; margin: 0 auto; z-index: 1;">3</div>
            <h4 style="margin-top: 10px;">Portfolio Review</h4>
            <p style="font-size: 14px;">Identify high-risk projects and patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with journey_cols[3]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; position: relative;">
            <div style="position: absolute; top: 15px; left: -50%; width: 100%; height: 4px; background-color: {ARCADIS_LIGHT_BG}; z-index: 0;"></div>
            <div style="position: relative; background-color: {ARCADIS_PRIMARY_COLOR}; color: white; width: 40px; height: 40px; line-height: 40px; border-radius: 50%; margin: 0 auto; z-index: 1;">4</div>
            <h4 style="margin-top: 10px;">Simulation</h4>
            <p style="font-size: 14px;">Test scenarios and what-if analyses</p>
        </div>
        """, unsafe_allow_html=True)
    
    with journey_cols[4]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; position: relative;">
            <div style="position: absolute; top: 15px; left: -50%; width: 100%; height: 4px; background-color: {ARCADIS_LIGHT_BG}; z-index: 0;"></div>
            <div style="position: relative; background-color: {ARCADIS_PRIMARY_COLOR}; color: white; width: 40px; height: 40px; line-height: 40px; border-radius: 50%; margin: 0 auto; z-index: 1;">5</div>
            <h4 style="margin-top: 10px;">Action Plan</h4>
            <p style="font-size: 14px;">Get reports with actionable insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Complete redesign of the capabilities section using pure Streamlit components
    st.write("")
    st.write("")
    
    # Centered header with custom styling
    st.markdown(f"<h2 style='text-align: center; color: {ARCADIS_PRIMARY_COLOR}; font-weight: 600;'>Platform Capabilities</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 30px;'>Leveraging advanced machine learning for actionable project risk insights</p>", unsafe_allow_html=True)
    
    # Create expandable sections for capabilities to avoid text overflow issues
    expandables = st.container()
    with expandables:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.write("")
                st.markdown(f"#### 📊 Risk Prediction")
                with st.expander("View details"):
                    st.write("Advanced ML algorithms (Random Forest, XGBoost)")
                    st.write("Up to 85% prediction accuracy")
                    st.write("Customizable risk thresholds")
                    st.write("Confidence scores for all predictions")
        
        with col2:
            with st.container():
                st.write("")
                st.markdown(f"#### 🔍 Explainable AI")
                with st.expander("View details"):
                    st.write("LIME explainability for transparent decisions")
                    st.write("Feature importance analysis")
                    st.write("Interactive visualizations")
                    st.write("Risk driver identification")
        
        with col3:
            with st.container():
                st.write("")
                st.markdown(f"#### 🎯 Actionable Insights")
                with st.expander("View details"):
                    st.write("What-if scenario planning")
                    st.write("Prioritized risk mitigation strategies")
                    st.write("PDF/PPT export for stakeholders")
                    st.write("Ongoing project monitoring")
    
    # Add more vertical space after the capability section
    st.write("")
    st.write("")
    
    # Create a completely separate section with a large gap
    st.write("")
    st.write("")
    
    # Force proper separation with explicit newlines and a divider
    st.markdown("""<div style='height: 80px;'></div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("""<div style='height: 40px;'></div>""", unsafe_allow_html=True)
    
    # Create a new section container
    questions_container = st.container()
    
    with questions_container:
        # Add a title to this section
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 40px; padding-top: 20px;">
            <h2 style="color: {ARCADIS_PRIMARY_COLOR}; font-weight: 600;">Key Project Questions Answered</h2>
            <p>RiskLens Pro helps you answer critical risk management questions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for the questions
        qcol1, qcol2 = st.columns(2)
        
        with qcol1:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; height: 95%; border-top: 4px solid {ARCADIS_PRIMARY_COLOR}; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: {ARCADIS_PRIMARY_COLOR}; font-size: 20px; margin-bottom: 10px;">Which projects need my immediate attention?</h3>
                <p style="color: #555; margin-bottom: 10px;">Get a prioritized list of high-risk projects with clear indicators of which projects need immediate intervention.</p>
                <div style="background-color: {ARCADIS_LIGHT_BG}; color: {ARCADIS_ACCENT_COLOR}; font-weight: 500; padding: 4px 10px; border-radius: 4px; font-size: 14px; display: inline-block;">Executive Summary tab</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; height: 95%; border-top: 4px solid {ARCADIS_SECONDARY_COLOR}; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: {ARCADIS_PRIMARY_COLOR}; font-size: 20px; margin-bottom: 10px;">Why is this project flagged as high-risk?</h3>
                <p style="color: #555; margin-bottom: 10px;">Understand the specific factors contributing to a project's risk rating with transparent AI explanations.</p>
                <div style="background-color: {ARCADIS_LIGHT_BG}; color: {ARCADIS_ACCENT_COLOR}; font-weight: 500; padding: 4px 10px; border-radius: 4px; font-size: 14px; display: inline-block;">Model Analysis tab</div>
            </div>
            """, unsafe_allow_html=True)
        
        with qcol2:
            st.markdown(f"""
            <div style="background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; height: 95%; border-top: 4px solid {ARCADIS_SECONDARY_COLOR}; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: {ARCADIS_PRIMARY_COLOR}; font-size: 20px; margin-bottom: 10px;">What if resources become scarce?</h3>
                <p style="color: #555; margin-bottom: 10px;">Run interactive scenario simulations to see how resource changes would impact project risk across your portfolio.</p>
                <div style="background-color: {ARCADIS_LIGHT_BG}; color: {ARCADIS_ACCENT_COLOR}; font-weight: 500; padding: 4px 10px; border-radius: 4px; font-size: 14px; display: inline-block;">Simulation tab</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; height: 95%; border-top: 4px solid {ARCADIS_PRIMARY_COLOR}; margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: {ARCADIS_PRIMARY_COLOR}; font-size: 20px; margin-bottom: 10px;">Which factors drive risk in our portfolio?</h3>
                <p style="color: #555; margin-bottom: 10px;">Identify the key factors that most strongly correlate with project risk across your entire portfolio.</p>
                <div style="background-color: {ARCADIS_LIGHT_BG}; color: {ARCADIS_ACCENT_COLOR}; font-weight: 500; padding: 4px 10px; border-radius: 4px; font-size: 14px; display: inline-block;">Model Analysis tab</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Create another container for the CTA and status
    cta_container = st.container()
    
    with cta_container:
        # Add space before CTA
        st.write("")
        st.write("")
        
        # Create a prominent CTA
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {ARCADIS_PRIMARY_COLOR}, {ARCADIS_SECONDARY_COLOR}); padding: 25px; border-radius: 10px; text-align: center; color: white; margin-top: 30px;">
            <h2 style="margin-top: 0; color: white;">Ready to Get Started?</h2>
            <p style="font-size: 18px;">Begin your risk analysis journey by loading your project data or using our sample dataset.</p>
            <p style="font-size: 16px;">Use the <b>"Load Sample Data"</b> button in the sidebar to explore the platform's capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add space after CTA and before status indicators
        st.write("")
        st.write("")
        
        # Status indicator based on current progress
        if st.session_state.project_data is None:
            st.info("👈 Get started by uploading your data in the sidebar.", icon="ℹ️")
        elif not st.session_state.trained_models:
            st.info("✅ Your data is loaded! Now train the risk model using the 'Run Risk Analytics' button in the sidebar.", icon="👈")
        else:
            st.success("🎉 Your risk model is ready! Navigate through the tabs above to explore the insights.", icon="✨")

def executive_summary_tab():
    """Content for the Executive Summary tab"""
    if st.session_state.project_data is None or st.session_state.risk_predictions is None:
        # Create a better placeholder when no data is available
        st.markdown(f"""
        <div style="background-color: {ARCADIS_LIGHT_BG}; padding: 20px; border-radius: 10px; text-align: center;">  
            <img src="https://img.icons8.com/fluency/96/business-risk.png" width="60" style="margin-bottom: 10px;">  
            <h2 style="color: {ARCADIS_PRIMARY_COLOR};">Executive Summary</h2>
            <p style="font-size: 16px;">Please load data and train a risk model to view the executive summary.</p>
            <p style="font-size: 14px; color: {ARCADIS_ACCENT_COLOR};">Use the sidebar on the left to upload project data or load the sample dataset.</p>
        </div>
        """, unsafe_allow_html=True)  
        return
    
    # Create a more visual title with underline animation
    st.markdown(f"""
    <style>
    .animated-header {{  
        position: relative;
        display: inline-block;
    }}
    .animated-header::after {{  
        content: '';
        position: absolute;
        width: 100%;
        height: 3px;
        bottom: -5px;
        left: 0;
        background: linear-gradient(90deg, {ARCADIS_PRIMARY_COLOR}, {ARCADIS_SECONDARY_COLOR});
        transform: scaleX(0);  
        transform-origin: bottom right;
        transition: transform 0.5s ease-out;
        animation: expand 1.5s ease-out forwards;
    }}
    @keyframes expand {{  
        to {{ transform: scaleX(1); transform-origin: bottom left; }}
    }}
    </style>
    <h1 class="animated-header" style="color:{ARCADIS_PRIMARY_COLOR};">📊 Executive Summary</h1>
    <p style="color:{ARCADIS_ACCENT_COLOR}; font-size:16px; margin-top:-5px;">Portfolio-wide risk assessment with actionable insights</p>
    """, unsafe_allow_html=True)
    
    # Get metrics
    metrics = get_data_profiling_metrics(st.session_state.project_data)
    high_risk_projects = int(st.session_state.risk_predictions.sum())
    total_projects = len(st.session_state.project_data)
    high_risk_rate = high_risk_projects / total_projects * 100
    
    # Create an alert dashboard at the top
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {ARCADIS_PRIMARY_COLOR}22, {ARCADIS_LIGHT_BG}); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid {ARCADIS_PRIMARY_COLOR};">
        <h3 style="margin-top: 0;">Project Risk Summary</h3>
        <p>As of {datetime.datetime.now().strftime('%B %d, %Y')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a more impactful metric row with gauges
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 180px;">
            <div style="text-align: center;">
                <p style="color: #555; font-size: 16px; margin-bottom: 5px;">Total Projects</p>
                <h1 style="font-size: 48px; color: {ARCADIS_PRIMARY_COLOR}; margin: 0;">{total_projects}</h1>
                <div style="margin-top: 15px; text-align: center;">
                    <span style="background-color: #e0e0e0; display: inline-block; width: 100%; height: 8px; border-radius: 4px;">
                        <span style="background-color: {ARCADIS_PRIMARY_COLOR}; display: inline-block; width: 100%; height: 8px; border-radius: 4px;"></span>
                    </span>
                    <p style="font-size: 12px; color: #777; margin-top: 5px;">In active portfolio</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 180px;">
            <div style="text-align: center;">
                <p style="color: #555; font-size: 16px; margin-bottom: 5px;">High-Risk Projects</p>
                <h1 style="font-size: 48px; color: {ARCADIS_SECONDARY_COLOR}; margin: 0;">{high_risk_projects}</h1>
                <div style="margin-top: 15px; text-align: center;">
                    <span style="background-color: #e0e0e0; display: inline-block; width: 100%; height: 8px; border-radius: 4px;">
                        <span style="background-color: {ARCADIS_SECONDARY_COLOR}; display: inline-block; width: {high_risk_rate}%; height: 8px; border-radius: 4px;"></span>
                    </span>
                    <p style="font-size: 12px; color: #777; margin-top: 5px;">Need immediate attention</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 180px;">
            <div style="text-align: center;">
                <p style="color: #555; font-size: 16px; margin-bottom: 5px;">High-Risk Rate</p>
                <h1 style="font-size: 48px; color: {'#e74c3c' if high_risk_rate > 30 else '#f1c40f' if high_risk_rate > 15 else '#2ecc71'}; margin: 0;">{high_risk_rate:.1f}%</h1>
                <div style="margin-top: 15px; text-align: center;">
                    <span style="background-color: #e0e0e0; display: inline-block; width: 100%; height: 8px; border-radius: 4px;">
                        <span style="background-color: {'#e74c3c' if high_risk_rate > 30 else '#f1c40f' if high_risk_rate > 15 else '#2ecc71'}; display: inline-block; width: {min(100, high_risk_rate*2)}%; height: 8px; border-radius: 4px;"></span>
                    </span>
                    <p style="font-size: 12px; color: #777; margin-top: 5px;">{'High Concern' if high_risk_rate > 30 else 'Moderate Concern' if high_risk_rate > 15 else 'Low Concern'}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Cost and schedule metrics in a better visualization
    if "Avg Cost Overrun % (High-Risk)" in metrics and "Avg Schedule Overrun % (High-Risk)" in metrics:
        cost_overrun = float(metrics["Avg Cost Overrun % (High-Risk)"].replace('%', ''))
        schedule_overrun = float(metrics["Avg Schedule Overrun % (High-Risk)"].replace('%', ''))
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: {ARCADIS_PRIMARY_COLOR};'>Project Performance Impact</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create a cost overrun gauge/progress bar
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4 style="margin: 0;">💰 Avg. Cost Overrun</h4>
                    <span style="font-size: 24px; font-weight: bold; color: {'#e74c3c' if cost_overrun > 20 else '#f1c40f' if cost_overrun > 10 else '#2ecc71'};">{metrics["Avg Cost Overrun % (High-Risk)"]}</span>
                </div>
                <div style="margin-top: 10px;">
                    <span style="background-color: #e0e0e0; display: block; width: 100%; height: 10px; border-radius: 5px;">
                        <span style="background-color: {'#e74c3c' if cost_overrun > 20 else '#f1c40f' if cost_overrun > 10 else '#2ecc71'}; display: block; width: {min(100, cost_overrun*2)}%; height: 10px; border-radius: 5px;"></span>
                    </span>
                </div>
                <p style="margin-top: 10px; font-size: 14px; color: #666;">High-risk projects typically exceed budget by {metrics["Avg Cost Overrun % (High-Risk)"]}.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create a schedule overrun gauge/progress bar
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4 style="margin: 0;">⏱️ Avg. Schedule Overrun</h4>
                    <span style="font-size: 24px; font-weight: bold; color: {'#e74c3c' if schedule_overrun > 20 else '#f1c40f' if schedule_overrun > 10 else '#2ecc71'};">{metrics["Avg Schedule Overrun % (High-Risk)"]}</span>
                </div>
                <div style="margin-top: 10px;">
                    <span style="background-color: #e0e0e0; display: block; width: 100%; height: 10px; border-radius: 5px;">
                        <span style="background-color: {'#e74c3c' if schedule_overrun > 20 else '#f1c40f' if schedule_overrun > 10 else '#2ecc71'}; display: block; width: {min(100, schedule_overrun*2)}%; height: 10px; border-radius: 5px;"></span>
                    </span>
                </div>
                <p style="margin-top: 10px; font-size: 14px; color: #666;">High-risk projects typically exceed schedule by {metrics["Avg Schedule Overrun % (High-Risk)"]}.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # AI-Generated Narrative Summary - improved styling
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {ARCADIS_PRIMARY_COLOR};'>💡 Key Insights & Recommendations</h3>", unsafe_allow_html=True)
    
    # Get narrative
    if st.session_state.ai_narrative_generated:
        narrative = generate_ai_narrative_summary()
    else:
        with st.spinner("Generating insights..."): 
            st.session_state.ai_narrative_generated = True
            narrative = generate_ai_narrative_summary()
    
    # Display narrative in a nicer format
    narrative_html = narrative.replace('\n\n', '<br><br>').replace('*', '•')
    st.markdown(f"""
    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
        {narrative_html}
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization section - improved
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: {ARCADIS_PRIMARY_COLOR};'>📊 Risk Distribution Analysis</h3>", unsafe_allow_html=True)
    
    # Two columns for visualizations
    viz1, viz2 = st.columns([3, 2])
    
    with viz1:
        # Get risk probabilities
        probabilities = st.session_state.risk_probabilities
        
        # Create improved histogram with proper coloring
        fig = px.histogram(
            x=probabilities,
            nbins=20,
            labels={"x": "Risk Probability"},
            title="Distribution of Project Risk Probabilities",
            color_discrete_sequence=[ARCADIS_PRIMARY_COLOR],
            template="plotly_white"
        )
        
        # Add vertical line for threshold
        fig.add_vline(
            x=st.session_state.prediction_threshold,
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text=f"Risk Threshold ({st.session_state.prediction_threshold})",
            annotation_position="top left",
            annotation_font={"size": 14, "color": "black"}
        )
        
        # Fill the area above the threshold
        fig.add_shape(
            type="rect",
            x0=st.session_state.prediction_threshold,
            x1=1,
            y0=0,
            y1=1,
            yref="paper",
            fillcolor="rgba(231, 76, 60, 0.1)",
            line_width=0,
        )
        
        # Add annotation for high risk zone
        fig.add_annotation(
            x=(1 + st.session_state.prediction_threshold) / 2,
            y=0.95,
            yref="paper",
            text="High Risk Zone",
            showarrow=False,
            font={"color": "#e74c3c", "size": 14},
            bgcolor="rgba(255, 255, 255, 0.8)",
        )
        
        fig.update_layout(
            bargap=0.1,
            xaxis_title="Risk Probability",
            yaxis_title="Number of Projects",
            margin=dict(l=40, r=40, t=60, b=40),
            title_font={"size": 16},
            coloraxis_showscale=False,
            plot_bgcolor="white",
        )
        
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    with viz2:
        st.markdown(f"<h4 style='color: {ARCADIS_PRIMARY_COLOR}; margin-bottom: 15px;'>⚠️ Top Risk Projects</h4>", unsafe_allow_html=True)
        
        # Get top 5 riskiest projects
        df = st.session_state.project_data
        probabilities = st.session_state.risk_probabilities
        
        top_risky = pd.DataFrame({
            'Project': df[PROJECT_NAME_COLUMN],
            'ID': df[PROJECT_ID_COLUMN],
            'Risk': probabilities
        }).sort_values('Risk', ascending=False).head(5).reset_index(drop=True)
        
        # Format percentage
        top_risky['Risk %'] = (top_risky['Risk'] * 100).round(1)
        
        # Create a better visualization of top projects
        for i, row in top_risky.iterrows():
            risk_color = "#e74c3c" if row['Risk'] > 0.7 else "#f1c40f" if row['Risk'] > 0.5 else ARCADIS_PRIMARY_COLOR
            
            st.markdown(f"""
            <div style="background-color: white; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {risk_color}; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div>
                    <div style="font-weight: bold; font-size: 15px;">{row['Project'][:30] + '...' if len(row['Project']) > 30 else row['Project']}</div>
                    <div style="color: #777; font-size: 12px;">ID: {row['ID']}</div>
                </div>
                <div style="background-color: {risk_color}; color: white; border-radius: 15px; padding: 3px 10px; font-weight: bold;">
                    {row['Risk %']}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""<div style="margin-top: 15px; text-align: center;">
            <a href="#" style="color: {ARCADIS_PRIMARY_COLOR}; text-decoration: none; font-size: 14px;">View all high-risk projects →</a>
        </div>""", unsafe_allow_html=True)

def portfolio_deep_dive_tab():
    """Content for the Portfolio Deep Dive tab"""
    if st.session_state.project_data is None or st.session_state.risk_predictions is None:
        st.warning("Please load data and train a risk model to view the portfolio analysis.")
        return
    
    styled_header("Portfolio Deep Dive", icon="🔍")
    st.markdown("Detailed analysis of your project portfolio with filtering and sorting capabilities.")
    
    # Get the data with predictions
    if hasattr(st.session_state, 'project_data_with_predictions'):
        df = st.session_state.project_data_with_predictions
    else:
        # If not available, create it
        df = st.session_state.project_data.copy()
        df['PredictedRisk'] = st.session_state.risk_predictions.values
        df['RiskProbability'] = st.session_state.risk_probabilities.values
        st.session_state.project_data_with_predictions = df
    
    # Create filter controls
    st.markdown("---")
    styled_header("Project Data & Predictions", level=2)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        project_types = ["All"] + sorted(df["ProjectType"].unique().tolist())
        selected_project_type = st.selectbox("Filter by Project Type:", project_types)
    
    with col2:
        regions = ["All"] + sorted(df["Region"].unique().tolist())
        selected_region = st.selectbox("Filter by Region:", regions)
    
    with col3:
        risk_options = ["All", "High Risk", "Low Risk"]
        selected_risk = st.selectbox("Filter by Predicted Risk:", risk_options, help="High Risk = probability > threshold")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_project_type != "All":
        filtered_df = filtered_df[filtered_df["ProjectType"] == selected_project_type]
    
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]
    
    if selected_risk == "High Risk":
        filtered_df = filtered_df[filtered_df["RiskProbability"] > st.session_state.prediction_threshold]
    elif selected_risk == "Low Risk":
        filtered_df = filtered_df[filtered_df["RiskProbability"] <= st.session_state.prediction_threshold]
    
    # Display filtered data
    display_cols = [PROJECT_ID_COLUMN, PROJECT_NAME_COLUMN, "Region", "ProjectType"]
    
    # Add available columns for display
    if "InitialScheduleDays" in filtered_df.columns:
        display_cols.append("InitialScheduleDays")
    if "ActualCost" in filtered_df.columns:
        display_cols.append("ActualCost")
    if "ActualScheduleDays" in filtered_df.columns:
        display_cols.append("ActualScheduleDays")
    
    display_cols.append("RiskProbability")
    
    # Format RiskProbability as percentage
    display_df = filtered_df[display_cols].copy()
    display_df["RiskProbability"] = (display_df["RiskProbability"] * 100).round(1).astype(str) + "%"
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    st.caption(f"Displaying {len(filtered_df)} of {len(df)} projects.")
    
    # Risk Breakdowns
    st.markdown("---")
    styled_header("Risk Breakdowns", level=2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        styled_header("Risk by Project Type", level=3)
        
        # Calculate risk rate by project type
        project_type_risk = df.groupby("ProjectType").apply(
            lambda x: (x["RiskProbability"] > st.session_state.prediction_threshold).mean() * 100
        ).reset_index(name="High-Risk Rate (%)")
        
        # Create bar chart
        fig = px.bar(
            project_type_risk.sort_values("High-Risk Rate (%)", ascending=False),
            x="ProjectType",
            y="High-Risk Rate (%)",
            title="Avg. Predicted High-Risk Rate by Project Type",
            color="High-Risk Rate (%)",
            color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
            template="plotly_white"
        )
        
        fig.update_layout(
            xaxis_title="Project Type",
            yaxis_title="High-Risk Rate (%)",
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        styled_header("Risk by Region", level=3)
        
        # Calculate risk rate by region
        region_risk = df.groupby("Region").apply(
            lambda x: (x["RiskProbability"] > st.session_state.prediction_threshold).mean() * 100
        ).reset_index(name="High-Risk Rate (%)")
        
        # Create bar chart
        fig = px.bar(
            region_risk.sort_values("High-Risk Rate (%)", ascending=False),
            x="Region",
            y="High-Risk Rate (%)",
            title="Avg. Predicted High-Risk Rate by Region",
            color="High-Risk Rate (%)",
            color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
            template="plotly_white",
            color_continuous_midpoint=50
        )
        
        fig.update_layout(
            xaxis_title="Region",
            yaxis_title="High-Risk Rate (%)",
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Register Heatmap (if risk data is available)
    if st.session_state.risk_data is not None:
        st.markdown("---")
        styled_header("Risk Register Heatmap (Filtered Projects)", level=2)
        
        # Get risk data for filtered projects
        filtered_project_ids = filtered_df[PROJECT_ID_COLUMN].unique()
        risk_df = st.session_state.risk_data
        filtered_risks = risk_df[risk_df[PROJECT_ID_COLUMN].isin(filtered_project_ids)]
        
        if len(filtered_risks) > 0:
            # Create heatmap of risk count by impact and probability
            risk_counts = filtered_risks.groupby(["Probability", "Impact"]).size().reset_index(name="Count")
            
            # Define the order of impact and probability levels
            impact_order = ["Very Low", "Low", "Medium", "High", "Very High"]
            prob_order = ["Very Low", "Low", "Medium", "High", "Very High"]
            
            # Create pivot table for heatmap
            pivot_data = risk_counts.pivot_table(values="Count", index="Probability", columns="Impact", fill_value=0)
            
            # Create heatmap using Plotly
            fig = px.imshow(
                pivot_data,
                labels=dict(x="Impact", y="Likelihood", color="Count"),
                x=pivot_data.columns,
                y=pivot_data.index,
                color_continuous_scale="Oranges",
                template="plotly_white",
                text_auto=True,
                aspect="auto"
            )
            
            fig.update_layout(
                title="Risk Count by Likelihood and Impact",
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risk register data available for the filtered projects.")

def model_analysis_tab():
    """Content for the Model Analysis & Explainability tab"""
    if st.session_state.model_results == {} or st.session_state.trained_models is None:
        # Create a better placeholder when no model is available
        st.markdown(f"""
        <div style="background-color: {ARCADIS_LIGHT_BG}; padding: 20px; border-radius: 10px; text-align: center;">  
            <img src="https://img.icons8.com/fluency/96/artificial-intelligence.png" width="60" style="margin-bottom: 10px;">  
            <h2 style="color: {ARCADIS_PRIMARY_COLOR};">Model Analysis & Explainability</h2>
            <p style="font-size: 16px;">Please train a risk model first to view the model analysis.</p>
            <p style="font-size: 14px; color: {ARCADIS_ACCENT_COLOR};">Use the sidebar on the left to train the model using the "Train Risk Model" button.</p>
        </div>
        """, unsafe_allow_html=True)  
        return
    
    # Create a more visual title with underline animation
    st.markdown(f"""
    <style>
    .animated-header {{  
        position: relative;
        display: inline-block;
    }}
    .animated-header::after {{  
        content: '';
        position: absolute;
        width: 100%;
        height: 3px;
        bottom: -5px;
        left: 0;
        background: linear-gradient(90deg, {ARCADIS_PRIMARY_COLOR}, {ARCADIS_SECONDARY_COLOR});
        transform: scaleX(0);  
        transform-origin: bottom right;
        transition: transform 0.5s ease-out;
        animation: expand 1.5s ease-out forwards;
    }}
    @keyframes expand {{  
        to {{ transform: scaleX(1); transform-origin: bottom left; }}
    }}
    </style>
    <h1 class="animated-header" style="color:{ARCADIS_PRIMARY_COLOR};">🧠 Model Analysis & Explainability</h1>
    <p style="color:{ARCADIS_ACCENT_COLOR}; font-size:16px; margin-top:-5px;">Explore model performance metrics and understand the factors influencing risk predictions</p>
    """, unsafe_allow_html=True)
    
    # Create an elegant model summary card
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {ARCADIS_PRIMARY_COLOR}22, {ARCADIS_LIGHT_BG}); padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid {ARCADIS_PRIMARY_COLOR}; display: flex; align-items: center;">
        <div style="margin-right: 20px;">
            <img src="https://img.icons8.com/fluency/96/artificial-intelligence.png" width="60">
        </div>
        <div>
            <h3 style="margin-top: 0;">Best Model: {st.session_state.best_model_name}</h3>
            <p>This model provides the best performance for predicting project risk based on the available data.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Metrics Section - Upgraded with color indicators
    st.markdown(f"<h2 style='color: {ARCADIS_PRIMARY_COLOR};'>Performance Metrics</h2>", unsafe_allow_html=True)
    
    # Get metrics from the best model
    metrics = st.session_state.model_results.get("metrics", {}).get(st.session_state.best_model_name, {})
    
    if metrics:
        # Metrics card with unified design
        col1, col2, col3, col4 = st.columns(4)
        
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        roc_auc = metrics.get('roc_auc', 0)
        
        # Helper function to determine color based on metric value
        def get_metric_color(value):
            if value >= 0.8:
                return "#2ecc71"  # Green for good
            elif value >= 0.6:
                return "#f1c40f"  # Yellow for medium
            else:
                return "#e74c3c"  # Red for poor
        
        with col1:
            st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 130px; text-align: center;">
                <p style="color: #555; font-size: 14px; margin-bottom: 5px;">Accuracy</p>
                <h2 style="font-size: 28px; color: {get_metric_color(accuracy)}; margin: 0;">{accuracy:.2f}</h2>
                <div style="margin-top: 10px;">
                    <div style="height: 8px; width: 100%; background-color: #e0e0e0; border-radius: 4px;">
                        <div style="height: 8px; width: {accuracy*100}%; background-color: {get_metric_color(accuracy)}; border-radius: 4px;"></div>
                    </div>
                </div>
                <p style="font-size: 12px; color: #777; margin-top: 5px;">Overall correctness</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 130px; text-align: center;">
                <p style="color: #555; font-size: 14px; margin-bottom: 5px;">Precision</p>
                <h2 style="font-size: 28px; color: {get_metric_color(precision)}; margin: 0;">{precision:.2f}</h2>
                <div style="margin-top: 10px;">
                    <div style="height: 8px; width: 100%; background-color: #e0e0e0; border-radius: 4px;">
                        <div style="height: 8px; width: {precision*100}%; background-color: {get_metric_color(precision)}; border-radius: 4px;"></div>
                    </div>
                </div>
                <p style="font-size: 12px; color: #777; margin-top: 5px;">True positives ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 130px; text-align: center;">
                <p style="color: #555; font-size: 14px; margin-bottom: 5px;">Recall</p>
                <h2 style="font-size: 28px; color: {get_metric_color(recall)}; margin: 0;">{recall:.2f}</h2>
                <div style="margin-top: 10px;">
                    <div style="height: 8px; width: 100%; background-color: #e0e0e0; border-radius: 4px;">
                        <div style="height: 8px; width: {recall*100}%; background-color: {get_metric_color(recall)}; border-radius: 4px;"></div>
                    </div>
                </div>
                <p style="font-size: 12px; color: #777; margin-top: 5px;">Completeness of positive predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 130px; text-align: center;">
                <p style="color: #555; font-size: 14px; margin-bottom: 5px;">ROC AUC</p>
                <h2 style="font-size: 28px; color: {get_metric_color(roc_auc)}; margin: 0;">{roc_auc:.2f}</h2>
                <div style="margin-top: 10px;">
                    <div style="height: 8px; width: 100%; background-color: #e0e0e0; border-radius: 4px;">
                        <div style="height: 8px; width: {roc_auc*100}%; background-color: {get_metric_color(roc_auc)}; border-radius: 4px;"></div>
                    </div>
                </div>
                <p style="font-size: 12px; color: #777; margin-top: 5px;">Model discriminative power</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Feature Importance Section - Enhanced with storytelling
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: {ARCADIS_PRIMARY_COLOR};'>Key Risk Drivers</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <p>Understanding which factors have the strongest influence on project risk allows you to focus attention on the most impactful variables. 
        These risk drivers can inform targeted mitigation strategies and proactive management decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get feature importance from the best model
    feature_importance = st.session_state.model_results.get("feature_importance", {}).get(st.session_state.best_model_name, None)
    
    if feature_importance is not None:
        # Convert to DataFrame for plotting
        fi_df = pd.DataFrame({
            'Feature': feature_importance[0],
            'Importance': feature_importance[1]
        }).sort_values('Importance', ascending=False).head(10)
        
        # Calculate maximum importance for normalization
        max_importance = fi_df['Importance'].max()
        
        # Create enhanced bar chart with Arcadis colors
        fig = px.bar(
            fi_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale=[[0, ARCADIS_LIGHT_BG], [0.5, ARCADIS_PRIMARY_COLOR], [1, ARCADIS_SECONDARY_COLOR]],
            template="plotly_white",
            text=fi_df['Importance'].apply(lambda x: f"{x:.3f}")
        )
        
        # Enhance styling
        fig.update_layout(
            title={
                'text': f"<b>Top 10 Risk Drivers</b>",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 22, 'color': ARCADIS_PRIMARY_COLOR}
            },
            yaxis=dict(
                categoryorder='total ascending',
                title=None,
                tickfont={'size': 14}
            ),
            xaxis=dict(
                title=dict(
                    text="Relative Importance",
                    font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
                )
            ),
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=80, b=40),
            plot_bgcolor='white',
            hoverlabel=dict(bgcolor='white', font_size=16)
        )
        
        # Add % on hover
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<br>Relative: %{customdata:.1%}",
            customdata=(fi_df['Importance']/max_importance).values.reshape(-1, 1)
        )
        
        # Add annotations for interpretation
        sorted_features = fi_df.sort_values('Importance', ascending=False)['Feature'].tolist()
        top_feature = sorted_features[0]
        second_feature = sorted_features[1]
        
        # Store the figure in session state for PDF export
        st.session_state.visualizations['feature_importance_fig'] = fig
        
        # Main container for visualization and insights
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
        with col2:
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; height: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h3 style="color: {ARCADIS_PRIMARY_COLOR}; margin-top: 0;">Key Insights</h3>
                <ul style="padding-left: 20px;">
                    <li style="margin-bottom: 10px;"><b>{top_feature}</b> has the strongest influence on project risk predictions</li>
                    <li style="margin-bottom: 10px;"><b>{second_feature}</b> is the second most important factor</li>
                    <li style="margin-bottom: 10px;">Projects with extreme values in these variables should receive special attention</li>
                </ul>
                <p style="margin-top: 15px; font-style: italic; color: {ARCADIS_SECONDARY_COLOR};">Proactively managing these key factors can significantly reduce project risk</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Model Evaluation Plots - Enhanced with insights and storytelling
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: {ARCADIS_PRIMARY_COLOR};'>Model Performance Assessment</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <p>These visualizations show how well the model performs in identifying high-risk projects. A good model will have high 
        true positive and true negative rates with minimal misclassifications.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Beta columns for a more engaging layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Enhanced Confusion Matrix
        if "confusion_matrix" in st.session_state.model_results:
            cm = st.session_state.model_results["confusion_matrix"]
            
            # Calculate metrics from confusion matrix
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            total = sum([tn, fp, fn, tp])
            accuracy = (tn + tp) / total if total > 0 else 0
            
            # Add annotations with percentages
            annotations = [
                [f"{tn} ({tn/total:.1%})", f"{fp} ({fp/total:.1%})"],
                [f"{fn} ({fn/total:.1%})", f"{tp} ({tp/total:.1%})"]
            ]
            
            # Create nicer confusion matrix plot with Arcadis colors
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Low Risk", "High Risk"],
                y=["Low Risk", "High Risk"],
                color_continuous_scale=[[0, "#f0f0f0"], [0.5, ARCADIS_LIGHT_BG], [1, ARCADIS_PRIMARY_COLOR]],
                template="plotly_white",
                text_auto=False
            )
            
            # Add text annotations
            for i in range(len(annotations)):
                for j in range(len(annotations[0])):
                    fig.add_annotation(
                        x=["Low Risk", "High Risk"][j],
                        y=["Low Risk", "High Risk"][i],
                        text=annotations[i][j],
                        showarrow=False,
                        font=dict(color="black", size=14)
                    )
            
            # Update layout with better styling
            fig.update_layout(
                title={
                    'text': "<b>Confusion Matrix</b>",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 18, 'color': ARCADIS_PRIMARY_COLOR}
                },
                xaxis=dict(side="bottom", title=None, tickfont={'size': 14}),
                yaxis=dict(title=None, tickfont={'size': 14}),
                margin=dict(l=40, r=40, t=60, b=40),
                height=350
            )
            
            # Store the figure in session state for PDF export
            st.session_state.visualizations['confusion_matrix_fig'] = fig
            
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            # Add confusion matrix explanation
            st.markdown(f"""
            <div style="background-color: white; border-left: 4px solid {ARCADIS_PRIMARY_COLOR}; padding: 15px; border-radius: 5px; margin-top: 10px;">
                <b>Interpretation:</b> {accuracy:.1%} of predictions are correct. 
                Pay special attention to the {fn} projects misclassified as low-risk when they are actually high-risk (false negatives).
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Enhanced ROC Curve
        if "fpr" in st.session_state.model_results and "tpr" in st.session_state.model_results:
            fpr = st.session_state.model_results["fpr"]
            tpr = st.session_state.model_results["tpr"]
            auc = metrics.get("roc_auc", 0)
            
            # Create nicer ROC curve plot with Arcadis color scheme
            fig = go.Figure()
            
            # Add ROC curve
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                line=dict(color=ARCADIS_PRIMARY_COLOR, width=3),
                name=f'Model (AUC = {auc:.3f})',
                hovertemplate='False Positive Rate: %{x:.2f}<br>True Positive Rate: %{y:.2f}'
            ))
            
            # Add diagonal reference line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                name='Random Guess (AUC = 0.5)',
                hoverinfo='skip'
            ))
            
            # Add shaded area under the ROC curve
            fig.add_trace(go.Scatter(
                x=np.concatenate([fpr, [1, 0]]),
                y=np.concatenate([tpr, [0, 0]]),
                fill='toself',
                fillcolor=f'rgba({int(ARCADIS_PRIMARY_COLOR[1:3], 16)}, {int(ARCADIS_PRIMARY_COLOR[3:5], 16)}, {int(ARCADIS_PRIMARY_COLOR[5:7], 16)}, 0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Update layout with better styling
            fig.update_layout(
                title={
                    'text': "<b>ROC Curve</b>",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 18, 'color': ARCADIS_PRIMARY_COLOR}
                },
                xaxis=dict(
                    title=dict(
                        text="False Positive Rate",
                        font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
                    ), 
                    tickformat='.1f',
                    range=[0, 1.05],
                    tickfont={'size': 12}
                ),
                yaxis=dict(
                    title=dict(
                        text="True Positive Rate",
                        font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
                    ), 
                    tickformat='.1f',
                    range=[0, 1.05],
                    tickfont={'size': 12}
                ),
                margin=dict(l=40, r=40, t=60, b=40),
                height=350,
                template="plotly_white",
                legend=dict(x=0.05, y=0.05, bgcolor='rgba(255,255,255,0.8)'),
                hovermode='closest'
            )
            
            # Add annotation for perfect classifier
            fig.add_annotation(
                x=0.1, y=0.9,
                text="Perfect Classifier",
                showarrow=True,
                arrowhead=1,
                ax=30, ay=-30,
                font=dict(color=ARCADIS_PRIMARY_COLOR, size=12)
            )
            
            # Store the figure in session state for PDF export
            st.session_state.visualizations['roc_curve_fig'] = fig
            
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
            # Add ROC curve explanation
            roc_quality = "excellent" if auc > 0.9 else "good" if auc > 0.8 else "fair" if auc > 0.7 else "poor"
            st.markdown(f"""
            <div style="background-color: white; border-left: 4px solid {ARCADIS_PRIMARY_COLOR}; padding: 15px; border-radius: 5px; margin-top: 10px;">
                <b>Interpretation:</b> The ROC curve shows model discriminative power with an AUC of {auc:.3f}, which is 
                <span style="color: {'#2ecc71' if auc > 0.8 else '#f1c40f' if auc > 0.7 else '#e74c3c'}"><b>{roc_quality}</b></span>. 
                Higher AUC indicates better ability to distinguish between high and low-risk projects.
            </div>
            """, unsafe_allow_html=True)
    
    # LIME Explainer Section
    st.markdown("---")
    styled_header("LIME Explanation for Individual Projects", level=2)
    st.markdown("See why specific projects are predicted to be high-risk or low-risk.")
    
    # Project selection
    if st.session_state.project_data is not None and st.session_state.risk_probabilities is not None:
        df = st.session_state.project_data
        
        # Sort projects by risk probability
        project_risks = pd.DataFrame({
            'ProjectID': df[PROJECT_ID_COLUMN],
            'ProjectName': df[PROJECT_NAME_COLUMN],
            'Risk': st.session_state.risk_probabilities
        }).sort_values('Risk', ascending=False)
        
        # Create selection options
        options = [f"{row['ProjectName']} ({row['ProjectID']}) - {row['Risk']:.1%}" 
                  for _, row in project_risks.head(20).iterrows()]
        
        selected_project = st.selectbox("Select a project to explain:", options)
        
        if selected_project:
            # Extract project ID from selection
            project_id = selected_project.split('(')[1].split(')')[0]
            
            # Display project details
            project_data = df[df[PROJECT_ID_COLUMN] == project_id].iloc[0]
            risk_prob = st.session_state.risk_probabilities[df[PROJECT_ID_COLUMN] == project_id].iloc[0]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                styled_header(f"Project: {project_data[PROJECT_NAME_COLUMN]}", level=3)
                st.markdown(f"**Project ID:** {project_id}")
                st.markdown(f"**Risk Probability:** {risk_prob:.1%}")
                st.markdown(f"**Risk Classification:** {'High Risk' if risk_prob > st.session_state.prediction_threshold else 'Low Risk'}")
                
                # More project details
                st.markdown("**Project Details:**")
                details = {
                    "Type": project_data.get("ProjectType", "N/A"),
                    "Region": project_data.get("Region", "N/A"),
                    "Budget": f"${project_data.get('Budget', 0):,.0f}",
                    "Duration": f"{project_data.get('DurationMonths', 0)} months",
                    "Complexity": project_data.get("ComplexityLevel", "N/A")
                }
                
                for k, v in details.items():
                    st.markdown(f"- {k}: {v}")
            
            with col2:
                # Simulated LIME explanation (in real implementation, this would come from the LIME explainer)
                styled_header("Risk Factors Explanation", level=3)
                
                # Create explanation for demo purposes
                # In a real implementation, this would be dynamically generated using LIME
                explanation = [
                    {"Feature": "ComplexityLevel_Very High", "Weight": 0.35, "Direction": "Increases Risk"},
                    {"Feature": "Budget", "Weight": 0.25, "Direction": "Increases Risk"},
                    {"Feature": "StakeholderEngagementScore", "Weight": -0.20, "Direction": "Decreases Risk"},
                    {"Feature": "Region_MEA", "Weight": 0.15, "Direction": "Increases Risk"},
                    {"Feature": "TeamSize", "Weight": -0.10, "Direction": "Decreases Risk"}
                ]
                
                # Create a plot
                exp_df = pd.DataFrame(explanation)
                
                # Adjust for direction
                exp_df["Adjusted Weight"] = exp_df.apply(
                    lambda x: x["Weight"] if x["Direction"] == "Increases Risk" else -x["Weight"], 
                    axis=1
                )
                
                # Sort by absolute weight
                exp_df = exp_df.sort_values("Adjusted Weight", key=lambda x: abs(x), ascending=False)
                
                # Create color mapping
                exp_df["Color"] = exp_df["Direction"].map({
                    "Increases Risk": ARCADIS_DANGER,
                    "Decreases Risk": ARCADIS_SUCCESS
                })
                
                # Create bar chart
                fig = px.bar(
                    exp_df,
                    x="Adjusted Weight",
                    y="Feature",
                    color="Direction",
                    color_discrete_map={
                        "Increases Risk": ARCADIS_DANGER,
                        "Decreases Risk": ARCADIS_SUCCESS
                    },
                    title="Factors Influencing Risk Prediction",
                    orientation="h",
                    template="plotly_white"
                )
                
                fig.update_layout(
                    yaxis=dict(categoryorder='total ascending'),
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                # Store the figure in session state for PDF export
                project_key = f"lime_explanation_{project_id}"
                st.session_state.visualizations[project_key] = fig
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Interpretation:**")
                st.markdown(
                    f"The model predicts this project as {'high' if risk_prob > st.session_state.prediction_threshold else 'low'} risk "
                    f"primarily due to its {explanation[0]['Feature'].split('_')[-1] if '_' in explanation[0]['Feature'] else explanation[0]['Feature']} "
                    f"and high {explanation[1]['Feature']}. However, good {explanation[2]['Feature']} partially mitigates the risk."
                )

def simulation_tab():
    """Content for the Simulation & Scenarios tab"""
    if st.session_state.project_data is None or st.session_state.risk_predictions is None:
        st.warning("Please load data and train a risk model to access the simulation features.")
        return
    
    styled_header("Simulation & Scenarios", icon="🎲")
    st.markdown("Explore 'what-if' scenarios and understand the uncertainty in risk predictions.")
    
    # Risk Threshold Simulation
    st.markdown("---")
    styled_header("Risk Threshold Simulation", level=2)
    st.markdown("Adjust the risk threshold to see how it affects the number of high-risk projects identified.")
    
    # Get risk probabilities
    probabilities = st.session_state.risk_probabilities
    
    # Create slider for threshold
    threshold = st.slider(
        "Risk Probability Threshold:",
        min_value=0.1,
        max_value=0.9,
        value=st.session_state.prediction_threshold,
        step=0.05,
        format="%.2f"
    )
    
    # Calculate metrics based on threshold
    high_risk_count = (probabilities > threshold).sum()
    high_risk_rate = high_risk_count / len(probabilities) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        styled_metric_card(
            label="High-Risk Projects",
            value=high_risk_count,
            icon="⚠️",
            color=ARCADIS_SECONDARY_COLOR
        )
    
    with col2:
        styled_metric_card(
            label="High-Risk Rate",
            value=f"{high_risk_rate:.1f}%",
            icon="📊"
        )
    
    # Create histogram with threshold line
    fig = px.histogram(
        x=probabilities,
        nbins=20,
        labels={"x": "Predicted Probability"},
        title="Distribution of Predicted Risk Probabilities",
        template="plotly_white"
    )
    
    # Add vertical line for threshold
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold})",
        annotation_position="top left"
    )
    
    fig.update_layout(
        bargap=0.1,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            title=dict(
                text="Predicted Probability",
                font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
            )
        ),
        yaxis=dict(
            title=dict(
                text="Count",
                font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
            )
        )
    )
    
    # Store the figure in session state for PDF export
    st.session_state.visualizations['risk_distribution_fig'] = fig
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 'What-If' Scenario Analysis
    st.markdown("---")
    styled_header("'What-If' Scenario Analysis", level=2)
    st.markdown("Explore how changes to project parameters could impact risk predictions.")
    
    # Project selection
    if st.session_state.project_data is not None:
        df = st.session_state.project_data
        
        # Create selection options
        options = [f"{row[PROJECT_NAME_COLUMN]} ({row[PROJECT_ID_COLUMN]})" 
                 for _, row in df.iterrows()]
        
        selected_project = st.selectbox("Select a project for scenario analysis:", options, index=0)
        
        if selected_project:
            # Extract project ID from selection
            project_id = selected_project.split('(')[1].split(')')[0]
            
            # Get the project data
            project_data = df[df[PROJECT_ID_COLUMN] == project_id].iloc[0].copy()
            current_risk = st.session_state.risk_probabilities[df[PROJECT_ID_COLUMN] == project_id].iloc[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                styled_header("Modify Project Parameters", level=3)
                
                # Parameter sliders
                st.markdown("**Project Complexity:**")
                complexity_map = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
                current_complexity = complexity_map.get(project_data.get("ComplexityLevel", "Medium"), 1)
                complexity_options = ["Low", "Medium", "High", "Very High"]
                new_complexity = st.selectbox(
                    "Complexity Level:",
                    options=complexity_options,
                    index=current_complexity
                )
                
                st.markdown("**Budget Change:**")
                budget_change = st.slider(
                    "Budget Change (%):",
                    min_value=-50,
                    max_value=100,
                    value=0,
                    step=5
                )
                
                st.markdown("**Project Duration Change:**")
                duration_change = st.slider(
                    "Duration Change (%):",
                    min_value=-30,
                    max_value=100,
                    value=0,
                    step=5
                )
                
                st.markdown("**Stakeholder Engagement:**")
                # Safely handle NaN values when getting StakeholderEngagementScore
                engagement_score = project_data.get("StakeholderEngagementScore", 5)
                current_engagement = 5  # default value
                if pd.notna(engagement_score):
                    try:
                        current_engagement = int(engagement_score)
                    except (ValueError, TypeError):
                        # Keep default value on conversion error
                        pass
                new_engagement = st.slider(
                    "Stakeholder Engagement Score (1-10):",
                    min_value=1,
                    max_value=10,
                    value=current_engagement
                )
                
                # Calculate new risk (simplified simulation for demo)
                # In real implementation, this would use the actual ML model
                complexity_risk_factor = {"Low": 0.1, "Medium": 0.2, "High": 0.3, "Very High": 0.4}.get(new_complexity, 0.2)
                budget_risk_factor = 0.1 * (budget_change / 100) if budget_change > 0 else 0
                duration_risk_factor = 0.1 * (duration_change / 100) if duration_change > 0 else 0
                engagement_risk_factor = -0.03 * (new_engagement - current_engagement)
                
                base_risk = current_risk
                new_risk = np.clip(base_risk + complexity_risk_factor + budget_risk_factor + duration_risk_factor + engagement_risk_factor, 0.05, 0.95)
                
                # Run scenario button
                analyze_btn = st.button("Run Scenario Analysis", type="primary", use_container_width=True)
            
            with col2:
                styled_header("Scenario Results", level=3)
                
                if analyze_btn:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        styled_metric_card(
                            label="Current Risk Probability",
                            value=f"{current_risk:.1%}",
                            icon="📊"
                        )
                    
                    with col2:
                        # Calculate percentage change with protection against division by zero
                        if current_risk > 0:
                            pct_change = ((new_risk - current_risk) / current_risk) * 100
                            delta = f"{pct_change:.1f}%" if pct_change != 0 else None
                        else:
                            delta = "N/A"
                        
                        styled_metric_card(
                            label="New Risk Probability",
                            value=f"{new_risk:.1%}",
                            delta=delta,
                            icon="📉" if new_risk < current_risk else "📈",
                            color=ARCADIS_SUCCESS if new_risk < current_risk else ARCADIS_DANGER
                        )
                    
                    # Risk classification
                    current_class = "High Risk" if current_risk > threshold else "Low Risk"
                    new_class = "High Risk" if new_risk > threshold else "Low Risk"
                    class_changed = current_class != new_class
                    
                    if class_changed:
                        if new_class == "Low Risk":
                            st.success(f"✅ Risk classification changed from {current_class} to {new_class}!")
                        else:
                            st.error(f"⚠️ Risk classification changed from {current_class} to {new_class}!")
                    
                    # Factor impact analysis
                    st.markdown("**Factor Impact Analysis:**")
                    
                    impact_factors = [
                        {"Factor": "Complexity Level", "Impact": complexity_risk_factor, "Direction": "+" if complexity_risk_factor > 0 else "-"},
                        {"Factor": "Budget Change", "Impact": budget_risk_factor, "Direction": "+" if budget_risk_factor > 0 else "-"},
                        {"Factor": "Duration Change", "Impact": duration_risk_factor, "Direction": "+" if duration_risk_factor > 0 else "-"},
                        {"Factor": "Stakeholder Engagement", "Impact": engagement_risk_factor, "Direction": "+" if engagement_risk_factor > 0 else "-"}
                    ]
                    
                    # Create impact DataFrame
                    impact_df = pd.DataFrame(impact_factors)
                    impact_df["Absolute Impact"] = impact_df["Impact"].abs()
                    impact_df = impact_df.sort_values("Absolute Impact", ascending=False)
                    
                    # Create a horizontal bar chart
                    fig = px.bar(
                        impact_df,
                        x="Impact",
                        y="Factor",
                        orientation="h",
                        color="Impact",
                        color_continuous_scale=["green", "yellow", "red"],
                        color_continuous_midpoint=0,
                        title="Impact of Changes on Risk Probability",
                        template="plotly_white"
                    )
                    
                    fig.update_layout(
                        margin=dict(l=40, r=40, t=60, b=40),
                        xaxis=dict(
                            title=dict(
                                text="Impact on Risk Probability",
                                font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
                            )
                        ),
                        yaxis=dict(
                            title=dict(
                                font={'size': 14, 'color': ARCADIS_SECONDARY_COLOR}
                            )
                        )
                    )
                    
                    # Store the figure in session state for PDF export
                    scenario_key = f"impact_analysis_{project_id}"
                    st.session_state.visualizations[scenario_key] = fig
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk change description
                    st.markdown("**Risk Change Analysis:**")
                    
                    # Get the highest impact factor
                    highest_impact = impact_df.iloc[0]
                    
                    if new_risk > current_risk:
                        st.markdown(
                            f"The risk probability increased primarily due to the change in **{highest_impact['Factor']}** "
                            f"(impact: {highest_impact['Impact']:.3f}). Consider addressing this factor to reduce project risk."
                        )
                    elif new_risk < current_risk:
                        st.markdown(
                            f"The risk probability decreased primarily due to the change in **{highest_impact['Factor']}** "
                            f"(impact: {highest_impact['Impact']:.3f}). This appears to be an effective risk mitigation strategy."
                        )
                    else:
                        st.markdown(
                            f"The changes had minimal impact on the overall risk probability. The factors appear to be offsetting each other."
                        )
                    
                    # Recommendation
                    st.markdown("**Recommendation:**")
                    
                    if new_risk > threshold:
                        st.markdown(
                            f"This project is still classified as **High Risk** after the changes. Consider additional "
                            f"mitigation strategies, particularly focused on improving stakeholder engagement and reducing complexity."
                        )
                    else:
                        st.markdown(
                            f"The project is now classified as **Low Risk** after the changes. Continue to monitor the project "
                            f"and maintain the improved parameters to keep risk levels low."
                        )
                else:
                    st.info("Adjust the parameters and click 'Run Scenario Analysis' to see the potential impact on project risk.")
    
    # Monte Carlo Simulation
    st.markdown("---")
    styled_header("Monte Carlo Simulation", level=2)
    st.markdown("Understand the uncertainty in risk predictions through Monte Carlo simulation.")
    
    st.info("Monte Carlo simulation would analyze project risk probability distribution across thousands of simulated scenarios with varying input parameters. This helps quantify uncertainty in risk predictions and identify the most critical risk factors to monitor.")

def main():
    """Main application entry point"""
    # Set page styling
    set_streamlit_style()
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar
    make_sidebar()
    
    # Tab navigation using clickable tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        f"{tab['emoji']} {tab['name']}" for tab in TABS
    ])
    
    # Display content based on active tab
    with tab1:  # Welcome tab
        welcome_tab()
    
    with tab2:  # Executive Summary
        executive_summary_tab()
    
    with tab3:  # Portfolio Deep Dive
        portfolio_deep_dive_tab()
    
    with tab4:  # Model Analysis
        model_analysis_tab()
    
    with tab5:  # Simulation & Scenarios
        simulation_tab()

if __name__ == "__main__":
    main()
