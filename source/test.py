import config
import load_data as ld
import data_preprocessing as dp
from pysurvival.utils import load_model
from pysurvival.utils.metrics import concordance_index

# ===================================================================================
# BEGINNING OF THE TESTING STEP - IMPORTANT TO VALIDATE THE MODEL
# ===================================================================================
# -----------------------------------------------------------------------------------
# Loads the data
# ==============
testing = ld.load_data(config.TESTING,config.ID_VAR)
# Imputes the missing values (NaNs)
# =================================
testing = dp.impute_test_data(testing,config.IMPUTED)
# Label encoding of categorical variables
# =======================================
testing = dp.categorical_encoding(testing,config.CAT_VARS)
# Normalization and scaling
# =========================
testing = dp.scaling_test_data(testing,config.FLOAT_VARS,config.SCALER)
# -----------------------------------------------------------------------------------
# ===================================================================================
# END OF THE TESTING STEP - IMPORTANT TO VALIDATE THE MODEL
# ===================================================================================


# ===================================================================================
# Scoring the model
# ===================================================================================
# -----------------------------------------------------------------------------------
model = load_model(config.MODEL)
X_test = testing.drop([config.TIMELINE,config.EVENT],axis=1)
T_test = testing[config.TIMELINE].values
E_test = testing[config.EVENT].values
C_index = concordance_index(model, X_test, T_test, E_test)
print("The concordance index on the test subsample is of: ",round(C_index,2)*100,'%')
# -----------------------------------------------------------------------------------
# ===================================================================================
# Scoring the model
# ===================================================================================