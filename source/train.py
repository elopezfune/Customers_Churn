import config, logging
import load_data as ld
import data_preprocessing as dp
import machine_learning_toolbox as ml
from pysurvival.utils.metrics import concordance_index

# Creating a logging file
logging.basicConfig(filename=config.SURV_LOGGING, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ===========================================================================================
# BEGINNING OF THE TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL
# ===========================================================================================
# -------------------------------------------------------------------------------------------
# Loads the data
# ==============
training = ld.load_data(config.TRAINING,config.ID_VAR)
logging.info('Data loaded correctly')

# Imputes the missing values (NaNs)
# =================================
training = dp.data_imputer(training,config.EVENT,config.IMPUTED)
logging.info('Missing values imputed')

# Label encoding of categorical variables
# =======================================
training = dp.categorical_encoding(training,config.CAT_VARS)
logging.info('All categorical variables are encoded')

# Normalization and scaling
# =========================
training = dp.standard_scaling(training,config.FLOAT_VARS,config.SCALER)
logging.info('All numerical variables are normalized and scaled')
# -------------------------------------------------------------------------------------------
# ===========================================================================================
# END OF THE TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL
# ===========================================================================================


# ===========================================================================================
# BEGINNING OF THE MODEL PRODUCTION
# ===========================================================================================
# -------------------------------------------------------------------------------------------
# Trains the survival model
# =========================
model = ml.conditional_survival_forest(training, config.TIMELINE, config.EVENT, config.MODEL)
logging.info('Model produced successfully')
# -------------------------------------------------------------------------------------------
# ===========================================================================================
# END OF THE MODEL PRODUCTION
# ===========================================================================================


# ===========================================================================================
# Internal scoring of the model
# ===========================================================================================
# -------------------------------------------------------------------------------------------
X_train = training.drop([config.TIMELINE,config.EVENT],axis=1)
T_train = training[config.TIMELINE].values
E_train = training[config.EVENT].values
C_index = concordance_index(model, X_train, T_train, E_train)
logging.info("The C-index of the model is: "+str(round(C_index,2)*100)+'%')