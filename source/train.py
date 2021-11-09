import config
import load_data as ld
import data_preprocessing as dp

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Loads the data
# ==============
training = ld.load_data(config.TRAINING,config.ID_VAR)

# Reduces memory usage
# ====================
training = dp.reduce_mem_usage(training)

# Deletes the duplicated rows
# ===========================
training = dp.duplicated_data(training)

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
#training = dp.standard_scaling(training,[config.TIMELINE, config.EVENT],config.SCALER)

# Reduces memory usage again
# ==========================
training = dp.reduce_mem_usage(training)

# Outliers removal
# ================
#training = dp.outlier_detector(training)

# Trains the survival model
# =========================
dp.conditional_survival_forest(training, config.TIMELINE, config.EVENT, config.MODEL)