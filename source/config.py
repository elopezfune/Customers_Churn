from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent

# Data Path
# =========
TRAINING = PROJECT / "data/training/training.json"
TESTING  = PROJECT / "data/testing/testing.json"
PREPROCESSED  = PROJECT / "data/preprocessed/preprocessed.csv"

# Variables
# =========
ID_VAR = "Client"

# Targets
TIMELINE = 'Tenure'
EVENT = 'Churn'

# Predictors
CAT_VARS = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod']
FLOAT_VARS = 'MonthlyCharges'



# Model Paths
# ===========
# Scaler path
SCALER = PROJECT / 'scaler/scaler.pkl'
# Survival model path
MODEL = PROJECT / 'models/csf_model.zip'