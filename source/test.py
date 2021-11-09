import config
import load_data as ld
import data_preprocessing as dp

# ================================================
# TESTING STEP - IMPORTANT TO VALIDATE THE MODEL

# Loads the data
# ==============
testing = ld.load_data(config.TESTING,config.ID_VAR)

# Reduces memory usage
# ====================
testing = dp.reduce_mem_usage(testing)

# Deletes the duplicated rows
# ===========================
testing = dp.duplicated_data(testing)

# Imputes variables with missing values (NaNs)
# ============================================
training = dp.data_imputer(training,config.EVENT)
        
# Label encoding of categorical variables
# =======================================
training = dp.categorical_encoding(training)

# Saving preprocessed data
# ========================
ld.save_csv(training,config.PREPROCESSED)

# Removing buggy variables
# ========================
training = training.drop('TotalCharges',axis=1)


# Normalization and scaling
# =========================
training = dp.standard_scaling(training,[config.TIMELINE, config.EVENT],config.SCALER)

# Reduces memory usage again
# ==========================
training = dp.reduce_mem_usage(training)

# Outliers removal
# ================
#training = dp.outlier_detector(training)

# Trains the survival model
# =========================
dp.conditional_survival_forest(training, config.TIMELINE, config.EVENT, config.MODEL)
