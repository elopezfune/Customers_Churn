from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
SURV_LOGGING = PROJECT / "loggins/training_survival_model.log"

# Data Path
# =========
TRAINING = PROJECT / "data/training/training.json"
TESTING  = PROJECT / "data/testing/testing.json"
IMPUTED  = PROJECT / "data/imputed/imputed.csv"


# Variables
# =========
ID_VAR = "Client"

# Predictors
CAT_VARS = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod']
FLOAT_VARS = ['MonthlyCharges','TotalCharges']

# Targets for Survival
TIMELINE = 'Tenure'
EVENT = 'Churn'


# Model Paths
# ===========
# Scaler path
SCALER = PROJECT / 'scaler/scaler.pkl'
# Survival model path
MODEL = PROJECT / 'models/csf_model.zip'