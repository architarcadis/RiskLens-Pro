"""
Data processing module for RiskLens Pro
Handles data import, validation, transformation and preparation
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import io
import os

# Constants
TARGET_VARIABLE = "ProjectDerailmentRisk"
PROJECT_ID_COLUMN = "ProjectID"
PROJECT_NAME_COLUMN = "ProjectName"

# Default features
DEFAULT_CATEGORICAL_FEATURES = [
    "ProjectType", "Region", "Sector", "ComplexityLevel", "ClientType"
]

DEFAULT_NUMERICAL_FEATURES = [
    "Budget", "DurationMonths", "TeamSize", "InitialRiskScore", 
    "ChangeRequests", "StakeholderEngagementScore"
]

# Column name patterns for semantic matching
column_patterns = {
    'project_id': ['id', 'project id', 'projectid', 'proj id', 'project_id', 'project #', 'project number'],
    'project_name': ['name', 'project name', 'projectname', 'title', 'project title', 'description', 'project description'],
    'budget': ['budget', 'cost', 'project cost', 'total cost', 'estimated cost', 'planned cost', 'project budget', 'financial'],
    'duration': ['duration', 'months', 'timeline', 'project duration', 'timeframe', 'time', 'schedule', 'weeks', 'days', 'length', 'plan duration'],
    'team_size': ['team', 'team size', 'staff', 'staff count', 'headcount', 'resources', 'team members', 'personnel', 'people', 'fte', 'employees'],
    'risk_score': ['risk', 'initial risk', 'risk score', 'risk assessment', 'risk rating', 'risk level', 'risk value', 'initial risk score'],
    'change_requests': ['change', 'changes', 'change requests', 'modifications', 'scope changes', 'requirement changes', 'cr count', 'crs', 'change orders'],
    'stakeholder_score': ['stakeholder', 'stakeholders', 'stakeholder score', 'stakeholder engagement', 'engagement', 'stakeholder rating', 'stakeholder management'],
    'project_type': ['type', 'project type', 'category', 'project category', 'classification', 'group', 'class'],
    'region': ['region', 'location', 'geography', 'area', 'country', 'territory', 'zone', 'market'],
    'sector': ['sector', 'industry', 'vertical', 'domain', 'business sector', 'market sector', 'field'],
    'complexity': ['complexity', 'complex', 'complexity level', 'difficulty', 'complexity score', 'technical complexity', 'complicated'],
    'client_type': ['client', 'client type', 'customer', 'customer type', 'account type', 'client category', 'customer category'],
    'target': ['risk', 'derailment', 'derailment risk', 'project derailment risk', 'project risk', 'success', 'failure', 'outcome', 'at risk', 'high risk', 'status']
}

@st.cache_data
def infer_data_columns(df):
    """
    Advanced column inference engine for RiskLens Pro
    Intelligently maps uploaded data columns to the required schema using
    semantic similarity, data type analysis, and value distribution patterns
    
    Args:
        df: DataFrame containing the uploaded data
    Returns:
        dict: Dictionary containing column category mappings and confidence scores
    """
    column_map = {}
    confidence_scores = {}
    
    # Use the module-level column_patterns for consistency
    
    # Analyze each column semantically and by data characteristics
    for col in df.columns:
        col_lower = col.lower()
        matches = {}
        
        # Check semantic matches with known patterns
        for key, patterns in column_patterns.items():
            for pattern in patterns:
                if pattern in col_lower or (len(pattern) > 4 and pattern.replace(' ', '') in col_lower.replace(' ', '')):
                    if key not in matches:
                        matches[key] = 0
                    # Higher score for more specific matches
                    if pattern == col_lower:
                        matches[key] += 1.0  # Exact match
                    elif pattern in col_lower:
                        matches[key] += 0.8  # Contains pattern
                    else:
                        matches[key] += 0.5  # Partial match
        
        # Analyze column data characteristics to refine matches
        if matches:
            # Data type refinement
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            is_categorical = pd.api.types.is_categorical_dtype(df[col]) or (
                pd.api.types.is_object_dtype(df[col]) and df[col].nunique() < len(df) * 0.5
            )
            is_unique = df[col].nunique() == len(df)
            high_cardinality = df[col].nunique() > 0.8 * len(df)
            
            # Value patterns
            contains_money = is_numeric and any('$' in str(x) for x in df[col].dropna().head(10))
            contains_percent = is_numeric and any('%' in str(x) for x in df[col].dropna().head(10))
            contains_dates = any(['date' in str(x).lower() for x in df[col].dropna().head(10)])
            
            # Remove inappropriate matches based on data characteristics
            keys_to_remove = []
            for key in matches:
                # Project ID should be unique
                if key == 'project_id' and not is_unique:
                    keys_to_remove.append(key)
                    
                # Budget, duration, team_size, etc. should be numeric
                elif key in ['budget', 'duration', 'team_size', 'risk_score', 'change_requests', 'stakeholder_score'] and not is_numeric:
                    keys_to_remove.append(key)
                    
                # Categorical features shouldn't be highly unique
                elif key in ['project_type', 'region', 'sector', 'complexity', 'client_type'] and high_cardinality:
                    keys_to_remove.append(key)
                    
                # Project name should have high cardinality
                elif key == 'project_name' and not high_cardinality:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del matches[key]
                
            # Adjust confidence based on data patterns
            for key in matches:
                # Budget confidence increased if contains currency symbols
                if key == 'budget' and contains_money:
                    matches[key] += 0.3
                    
                # Risk scores often contain percentages
                elif key == 'risk_score' and contains_percent:
                    matches[key] += 0.2
                    
                # Duration typically not in dates
                elif key == 'duration' and contains_dates:
                    matches[key] -= 0.3
                    
                # Target variable confidence adjusted based on distribution
                elif key == 'target':
                    # Ideal target is binary or has low cardinality
                    if df[col].nunique() <= 2:
                        matches[key] += 0.5
                    elif df[col].nunique() <= 5:
                        matches[key] += 0.2
                    
            # Select the best match for this column
            if matches:
                best_match = max(matches.items(), key=lambda x: x[1])
                if best_match[1] >= 0.5:  # Only keep if confidence is reasonable
                    # Store with column category and confidence
                    if best_match[0] not in column_map or matches[best_match[0]] > confidence_scores.get(best_match[0], 0):
                        column_map[best_match[0]] = col
                        confidence_scores[best_match[0]] = best_match[1]
    
    # Categorize remaining uncategorized columns based on data type
    numerical_features = []
    categorical_features = []
    
    for col in df.columns:
        if col not in column_map.values():  # Skip columns already mapped
            if pd.api.types.is_numeric_dtype(df[col]):
                numerical_features.append(col)
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                # Check if it's not a high cardinality column (like comments, descriptions)
                if df[col].nunique() < len(df) * 0.5:
                    categorical_features.append(col)
    
    # Add the categorized features to the map
    column_map['numerical_features'] = numerical_features
    column_map['categorical_features'] = categorical_features
    column_map['confidence_scores'] = confidence_scores
    
    return column_map

def create_preprocessing_pipeline(categorical_features, numerical_features):
    """
    Create a scikit-learn preprocessing pipeline for the data
    Args:
        categorical_features: List of categorical feature column names
        numerical_features: List of numerical feature column names
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    # Numerical features pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

def transform_data_to_template(df):
    """
    Advanced data transformation engine that intelligently maps uploaded data 
    to the required template format with detailed analysis of mapping quality
    
    Args:
        df: DataFrame containing the uploaded data
    Returns:
        tuple: (transformed_df, mapping_info, missing_data_impact)
    """
    # First, infer the column types with our advanced matching algorithm
    column_map = infer_data_columns(df)
    confidence_scores = column_map.get('confidence_scores', {})
    
    # Create a new DataFrame with the transformed data
    transformed_df = pd.DataFrame()
    mapping_info = {}
    missing_data_impact = {}
    
    # Set the project ID
    if 'project_id' in column_map:
        transformed_df[PROJECT_ID_COLUMN] = df[column_map['project_id']]
        mapping_info[PROJECT_ID_COLUMN] = {
            'source': column_map['project_id'],
            'confidence': confidence_scores.get('project_id', 0.8),
            'status': 'mapped'
        }
    else:
        # Generate project IDs if not found
        transformed_df[PROJECT_ID_COLUMN] = [f"PROJ{1000+i}" for i in range(len(df))]
        mapping_info[PROJECT_ID_COLUMN] = {
            'source': "Generated",
            'confidence': 1.0,  # High confidence since we generated it
            'status': 'generated'
        }
        # No real impact from missing project IDs as we can generate them
    
    # Set the project name
    if 'project_name' in column_map:
        transformed_df[PROJECT_NAME_COLUMN] = df[column_map['project_name']]
        mapping_info[PROJECT_NAME_COLUMN] = {
            'source': column_map['project_name'],
            'confidence': confidence_scores.get('project_name', 0.8),
            'status': 'mapped'
        }
    else:
        # Generate project names if not found
        transformed_df[PROJECT_NAME_COLUMN] = [f"Project {i+1}" for i in range(len(df))]
        mapping_info[PROJECT_NAME_COLUMN] = {
            'source': "Generated",
            'confidence': 1.0,
            'status': 'generated'
        }
        missing_data_impact['project_name'] = {
            'severity': 'low',
            'impact': 'Identifiability of projects will be limited to generic names',
            'recommendation': 'Consider adding descriptive project names for better identification'
        }
    
    # Key category mappings
    feature_importance = {
        # Categorical features - Key importance ratings and impacts
        'ProjectType': {
            'importance': 'high',
            'impact': 'Critical for categorizing projects and identifying patterns in similar project types',
            'recommendation': 'This is a primary factor in risk assessment, highly recommended to provide'
        },
        'Region': {
            'importance': 'medium',
            'impact': 'Geographic patterns affect risk profiles, limiting regional analysis',
            'recommendation': 'Provides context for regional performance benchmarking'
        },
        'Sector': {
            'importance': 'high',
            'impact': 'Different sectors show distinct risk patterns and success factors',
            'recommendation': 'This is a key factor in risk assessment models, highly recommended to provide'
        },
        'ComplexityLevel': {
            'importance': 'high',
            'impact': 'Project complexity is directly correlated with risk level',
            'recommendation': 'Critical for accurate risk assessment, provide if possible'
        },
        'ClientType': {
            'importance': 'medium',
            'impact': 'Client relationships affect project dynamics and risks',
            'recommendation': 'Useful for stakeholder analysis and communication strategies'
        },
        
        # Numerical features - Key importance ratings and impacts
        'Budget': {
            'importance': 'high',
            'impact': 'Financial scope is a key risk driver and affects analysis accuracy',
            'recommendation': 'Critical financial dimension, strongly recommended to provide'
        },
        'DurationMonths': {
            'importance': 'high',
            'impact': 'Project timeline is a primary factor in risk assessment',
            'recommendation': 'Time dimension is critical for risk modeling, strongly recommended'
        },
        'TeamSize': {
            'importance': 'medium',
            'impact': 'Resource allocation affects project execution capability',
            'recommendation': 'Provides context for resource-related risk factors'
        },
        'InitialRiskScore': {
            'importance': 'high',
            'impact': 'Initial risk assessment provides baseline for comparison',
            'recommendation': 'Very valuable for trend analysis and benchmark comparison'
        },
        'ChangeRequests': {
            'importance': 'medium',
            'impact': 'Scope changes correlate with project disruption',
            'recommendation': 'Useful indicator of project stability and management'
        },
        'StakeholderEngagementScore': {
            'importance': 'medium',
            'impact': 'Stakeholder engagement correlates with project success',
            'recommendation': 'Provides insight into organizational dynamics affecting projects'
        }
    }
    
    # Map categorical features with intelligent pattern matching
    for cat_feature in DEFAULT_CATEGORICAL_FEATURES:
        target_key = None
        for key in ['project_type', 'region', 'sector', 'complexity', 'client_type']:
            if key in cat_feature.lower() or cat_feature.lower() in key:
                target_key = key
                break
                
        if target_key and target_key in column_map:
            # Direct mapping found by semantic engine
            transformed_df[cat_feature] = df[column_map[target_key]]
            mapping_info[cat_feature] = {
                'source': column_map[target_key],
                'confidence': confidence_scores.get(target_key, 0.7),
                'status': 'mapped'
            }
        else:
            # Try additional pattern matching
            matching_cols = []
            
            # Try pattern match from column names
            for col in df.columns:
                if cat_feature.lower().replace('_', '') in col.lower().replace('_', '') or \
                   any(pattern in col.lower() for pattern in column_patterns.get(target_key, [])):
                    matching_cols.append(col)
            
            if matching_cols:
                # Take the best match based on similarity
                best_match = matching_cols[0]  # Default to first match
                transformed_df[cat_feature] = df[best_match]
                mapping_info[cat_feature] = {
                    'source': best_match,
                    'confidence': 0.6,  # Medium confidence for pattern matches
                    'status': 'pattern_matched'
                }
            else:
                # No suitable match found - use placeholder but record impact
                transformed_df[cat_feature] = "Unknown"
                mapping_info[cat_feature] = {
                    'source': "Missing",
                    'confidence': 1.0,  # High confidence that it's missing
                    'status': 'missing'
                }
                
                # Record the impact of this missing feature
                feature_info = feature_importance.get(cat_feature, {
                    'importance': 'medium',
                    'impact': f'Missing {cat_feature} limits categorical analysis',
                    'recommendation': f'Recommend providing {cat_feature} for better results'
                })
                
                missing_data_impact[cat_feature] = {
                    'severity': feature_info['importance'],
                    'impact': feature_info['impact'],
                    'recommendation': feature_info['recommendation']
                }
    
    # Map numerical features with similar intelligent approach
    for num_feature in DEFAULT_NUMERICAL_FEATURES:
        # Try to find direct mapping from our semantic engine
        target_key = None
        for key in ['budget', 'duration', 'team_size', 'risk_score', 'change_requests', 'stakeholder_score']:
            if key in num_feature.lower() or num_feature.lower() in key:
                target_key = key
                break
        
        if target_key and target_key in column_map:
            # Direct mapping found
            transformed_df[num_feature] = df[column_map[target_key]]
            mapping_info[num_feature] = {
                'source': column_map[target_key],
                'confidence': confidence_scores.get(target_key, 0.7),
                'status': 'mapped'
            }
        else:
            # Try additional pattern matching from column names
            matching_cols = []
            for col in df.columns:
                if num_feature.lower().replace('_', '') in col.lower().replace('_', '') and \
                   pd.api.types.is_numeric_dtype(df[col]):
                    matching_cols.append(col)
            
            if matching_cols:
                # Take the best match
                best_match = matching_cols[0]
                transformed_df[num_feature] = df[best_match]
                mapping_info[num_feature] = {
                    'source': best_match,
                    'confidence': 0.6,  # Medium confidence
                    'status': 'pattern_matched'
                }
            else:
                # No suitable match - use NaN but record impact
                transformed_df[num_feature] = np.nan
                mapping_info[num_feature] = {
                    'source': "Missing",
                    'confidence': 1.0,
                    'status': 'missing'
                }
                
                # Record the impact of this missing feature
                feature_info = feature_importance.get(num_feature, {
                    'importance': 'medium',
                    'impact': f'Missing {num_feature} reduces numerical analysis capability',
                    'recommendation': f'Recommend providing {num_feature} for better results'
                })
                
                missing_data_impact[num_feature] = {
                    'severity': feature_info['importance'],
                    'impact': feature_info['impact'],
                    'recommendation': feature_info['recommendation']
                }
    
    # Map target variable with special handling as it's critical
    if 'target' in column_map:
        target_col = column_map['target']
        target_values = df[target_col]
        
        # Check if binary
        if target_values.nunique() <= 2:
            # Convert to 0/1
            if not pd.api.types.is_numeric_dtype(target_values):
                # If categorical, map to 0/1
                pos_values = ['high', 'yes', 'true', '1', 'high risk', 'derailed', 'at risk', 'failed', 'fail']
                transformed_df[TARGET_VARIABLE] = target_values.apply(
                    lambda x: 1 if str(x).lower() in pos_values else 0
                )
            else:
                # If numeric, ensure it's 0/1
                transformed_df[TARGET_VARIABLE] = target_values.apply(
                    lambda x: 1 if x > 0 else 0
                )
                
            mapping_info[TARGET_VARIABLE] = {
                'source': target_col,
                'confidence': confidence_scores.get('target', 0.9),  # High confidence for binary target
                'status': 'mapped_binary'
            }
        else:
            # If not binary, convert to binary based on threshold
            # We'll check the data distribution to determine the threshold
            if pd.api.types.is_numeric_dtype(target_values):
                # For numeric, use upper quartile as the threshold
                threshold = target_values.quantile(0.75)  # Upper quartile
                transformed_df[TARGET_VARIABLE] = target_values.apply(
                    lambda x: 1 if x > threshold else 0
                )
                mapping_info[TARGET_VARIABLE] = {
                    'source': target_col,
                    'confidence': 0.7,  # Medium confidence for converted target
                    'status': 'converted_from_numeric',
                    'conversion_note': f'Converted to binary using threshold of {threshold}'
                }
            else:
                # For categorical, use most severe category as high risk
                severity_indicators = ['high', 'critical', 'severe', 'important', 'urgent', 'major']
                # Check if any values contain severity indicators
                high_severity_values = [val for val in target_values.unique() 
                                      if any(indicator in str(val).lower() for indicator in severity_indicators)]
                
                if high_severity_values:
                    transformed_df[TARGET_VARIABLE] = target_values.apply(
                        lambda x: 1 if x in high_severity_values else 0
                    )
                    mapping_info[TARGET_VARIABLE] = {
                        'source': target_col,
                        'confidence': 0.65,  # Medium-low confidence
                        'status': 'converted_from_categorical',
                        'conversion_note': f'Converted using severity indicators for {high_severity_values}'
                    }
                else:
                    # Last resort - take the least common value as the high risk
                    val_counts = target_values.value_counts()
                    least_common = val_counts.idxmin()
                    transformed_df[TARGET_VARIABLE] = target_values.apply(
                        lambda x: 1 if x == least_common else 0
                    )
                    mapping_info[TARGET_VARIABLE] = {
                        'source': target_col,
                        'confidence': 0.5,  # Low confidence
                        'status': 'converted_frequency_based',
                        'conversion_note': f'Converted assuming least common value ({least_common}) is high risk'
                    }
        
    else:
        # No target variable found - this is critical
        transformed_df[TARGET_VARIABLE] = np.nan  # We'll prompt user to provide this
        mapping_info[TARGET_VARIABLE] = {
            'source': "Missing",
            'confidence': 1.0,
            'status': 'missing_critical'
        }
        
        missing_data_impact[TARGET_VARIABLE] = {
            'severity': 'critical',
            'impact': 'Target variable is essential for risk prediction. Without it, supervised learning cannot be performed.',
            'recommendation': 'Please provide a risk indicator column such as "Project Risk", "Success/Failure", "Project Status", or any field indicating project outcome.'
        }
    
    # Calculate the overall data quality score
    quality_scores = []
    missing_core_features = []
    
    # Critical path - must have Project ID and Target Variable
    if mapping_info[PROJECT_ID_COLUMN]['status'] != 'missing':
        quality_scores.append(1.0)
    else:
        quality_scores.append(0.0)
        missing_core_features.append(PROJECT_ID_COLUMN)
    
    if mapping_info[TARGET_VARIABLE]['status'] not in ['missing', 'missing_critical']:
        quality_scores.append(1.0)
    else:
        quality_scores.append(0.0)
        missing_core_features.append(TARGET_VARIABLE)
    
    # For each categorical feature, score 0.8 if present, 0.0 if missing
    for feature in DEFAULT_CATEGORICAL_FEATURES:
        if mapping_info[feature]['status'] != 'missing':
            quality_scores.append(0.8)
        else:
            quality_scores.append(0.0)
            # Only track high importance missing features
            if feature_importance.get(feature, {}).get('importance') == 'high':
                missing_core_features.append(feature)
    
    # For each numerical feature, score 0.8 if present, 0.0 if missing
    for feature in DEFAULT_NUMERICAL_FEATURES:
        if mapping_info[feature]['status'] != 'missing':
            quality_scores.append(0.8)
        else:
            quality_scores.append(0.0)
            # Only track high importance missing features
            if feature_importance.get(feature, {}).get('importance') == 'high':
                missing_core_features.append(feature)
    
    # Average the scores
    data_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    # Compile overall impact assessment
    overall_impact = {
        'data_quality_score': data_quality_score,
        'missing_core_features': missing_core_features,
        'total_features': len(DEFAULT_CATEGORICAL_FEATURES) + len(DEFAULT_NUMERICAL_FEATURES) + 2,  # +2 for ID and Target
        'mapped_features': sum(1 for score in quality_scores if score > 0),
        'analysis_capability': 'full' if data_quality_score > 0.8 else 
                              'limited' if data_quality_score > 0.5 else 
                              'severely_limited' if data_quality_score > 0.3 else 'not_possible'
    }
    
    # Add overall impact to the missing_data_impact dictionary
    missing_data_impact['overall'] = overall_impact
    
    # No longer needed here, using the column_patterns defined at module level
    
    return transformed_df, mapping_info, missing_data_impact

def validate_data(df):
    """
    Validate the data for required columns and formats
    Args:
        df: DataFrame to validate
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check for minimum number of rows
    if len(df) < 5:
        return False, f"Dataset has only {len(df)} rows. At least 5 rows are required."
    
    # Check for ProjectID column
    if PROJECT_ID_COLUMN not in df.columns:
        return False, f"Required column '{PROJECT_ID_COLUMN}' is missing."
    
    # Check for unique ProjectIDs
    if df[PROJECT_ID_COLUMN].duplicated().any():
        return False, f"Column '{PROJECT_ID_COLUMN}' contains duplicate values."
    
    # Check for target variable
    if TARGET_VARIABLE not in df.columns:
        return False, f"Target variable '{TARGET_VARIABLE}' is missing."
    
    # Check for numerical features
    num_features_present = [col for col in DEFAULT_NUMERICAL_FEATURES if col in df.columns]
    if len(num_features_present) < 2:
        return False, f"At least 2 numerical features are required. Found {len(num_features_present)}."
    
    # Check for categorical features
    cat_features_present = [col for col in DEFAULT_CATEGORICAL_FEATURES if col in df.columns]
    if len(cat_features_present) < 1:
        return False, f"At least 1 categorical feature is required. Found {len(cat_features_present)}."
    
    # Check for minimum non-null values
    null_counts = df.isnull().sum()
    for col in df.columns:
        if null_counts[col] > 0.5 * len(df):
            return False, f"Column '{col}' has more than 50% missing values."
    
    return True, "Data validation successful."

def get_column_statistics(df):
    """
    Generate summary statistics for each column in the dataframe
    Args:
        df: DataFrame to analyze
    Returns:
        dict: Dictionary of column statistics
    """
    stats = {}
    
    for col in df.columns:
        col_stats = {}
        
        # Basic info
        col_stats['dtype'] = str(df[col].dtype)
        col_stats['count'] = len(df[col])
        col_stats['missing'] = df[col].isnull().sum()
        col_stats['missing_percent'] = (col_stats['missing'] / col_stats['count']) * 100
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric column stats
            col_stats['min'] = df[col].min()
            col_stats['max'] = df[col].max()
            col_stats['mean'] = df[col].mean()
            col_stats['median'] = df[col].median()
            col_stats['std'] = df[col].std()
        else:
            # Categorical column stats
            col_stats['unique_values'] = df[col].nunique()
            if col_stats['unique_values'] <= 10:
                col_stats['value_counts'] = df[col].value_counts().to_dict()
        
        stats[col] = col_stats
    
    return stats

def handle_file_upload(uploaded_file):
    """
    Process uploaded file and return DataFrame
    Args:
        uploaded_file: Streamlit uploaded file object
    Returns:
        DataFrame or None if processing failed
    """
    try:
        # Get file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Load data based on file type
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}. Please upload a CSV or Excel file.")
            return None
        
        # Basic data cleaning
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Remove completely empty rows
        df = df.dropna(axis=0, how='all')
        
        # Convert column names to strings and strip whitespace
        df.columns = df.columns.astype(str).str.strip()
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def split_train_test_data(df, target_col, test_size=0.25, random_state=42):
    """
    Split data into training and testing sets
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Split into features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Also drop project ID and name columns if they exist
    if PROJECT_ID_COLUMN in X.columns:
        X = X.drop(columns=[PROJECT_ID_COLUMN])
    
    if PROJECT_NAME_COLUMN in X.columns:
        X = X.drop(columns=[PROJECT_NAME_COLUMN])
    
    # Split into train and test sets
    # Try using stratified split, but fall back to a regular split if there's not enough data in each class
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError as e:
        # If stratification fails, fall back to a regular split without stratification
        print(f"Warning: Stratified split failed, using regular split instead. Error: {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
    
    return X_train, X_test, y_train, y_test
