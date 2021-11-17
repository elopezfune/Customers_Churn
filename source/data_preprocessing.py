# Data prepocessing module
# ========================

# Standard libraries
# ==================
import joblib
import numpy as np
import pandas as pd


# Preprocessing libraries
# =======================
from sklearn.preprocessing import MinMaxScaler


# Neglect warnings
# ================
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# =======================================================================================================
# BEGINNING OF THE PREPROCESSING FOR  THE TRAINING SUBSAMPLE
# =======================================================================================================
# -------------------------------------------------------------------------------------------------------

# =======================================================================================================
# Reduces memory usage in data
# =======================================================================================================
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    # Copies the dataframe
    df = df.copy()
    
    # Computes the initial memory usage after loading the data
    start_mem = df.memory_usage().sum() / 1024**2
    
    # Does a loop by all the columns to convert to other data type 
    for el in df.columns:
        col_type = df[el].dtype
        
        if col_type != object:
            c_min = df[el].min()
            c_max = df[el].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[el] = df[el].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[el] = df[el].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[el] = df[el].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[el] = df[el].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[el] = df[el].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[el] = df[el].astype(np.float32)
                else:
                    df[el] = df[el].astype(np.float64)
        else:
            df[el] = df[el].astype('category')
    
    # Computes the memory usage after these transformations 
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    # Returns the reduced-memory dataframe
    return df
# =======================================================================================================
# Reduces memory usage in data
# =======================================================================================================

# =======================================================================================================
# Checks for duplicated data and removes them
# =======================================================================================================
def duplicated_data(df):
    # Copies the dataframe
    df = df.copy()
    # Rows containing duplicate data
    print("Removed ", df[df.duplicated()].shape[0], ' duplicate rows')
    # Returns a dataframe with the duplicated rows removed
    return df.drop_duplicates()
# =======================================================================================================
# Checks for duplicated data and removes them
# =======================================================================================================



# =======================================================================================================
# Missing values imputation
# =======================================================================================================
# Checks for columns with missing values (NaNs)
def check_missing_values(df,cols=None,axis=0):
    # Copies the dataframe
    df = df.copy()
    if cols != None:
        df = df[cols]
    missing_num = df.isnull().sum(axis).to_frame().rename(columns={0:'missing_num'})
    missing_num['missing_percent'] = df.isnull().mean(axis)*100
    result = missing_num.sort_values(by='missing_percent',ascending = False)
    # Returns a dataframe with columns with missing data as index and the number and percent of NaNs
    return result[result["missing_percent"]>0.0]

# Missing values imputer
def data_imputer(df,target,output_path):
    # Copies the dataframe
    df = df.copy()
    # Checks for missing values
    missing = check_missing_values(df)
    if len(missing)!=0:
        for el in missing.index:
            if df[el].dtype.kind in 'biufc':
                # Imputes the missing values of a numerical variable with the median
                df[el] = df[[el,target]].groupby(str(target)).transform(lambda x: x.fillna(x.median()))
            else:
                # Imputes the missing values of a categorial or string variable with the mode
                df[el] = df[[el,target]].groupby(str(target)).transform(lambda x: x.fillna(x.mode()[0]))
    #print('Missing values imputed')
    del missing
    # Saves and returns the dataframe with the missing values imputed
    df.to_csv(output_path)
    return df
# =======================================================================================================
# Missing values imputation
# =======================================================================================================


# =======================================================================================================
# Categorical variable encoding
# =======================================================================================================
def categorical_encoding(df,cat_vars,drop_first=False):
    # Copies the fataframe
    df = df.copy()
    # Encodes the categorical
    df_to_encode = df[cat_vars]
    df_encoded = pd.get_dummies(df_to_encode,drop_first=drop_first)
    #Formats the names of the variables
    df_encoded.columns = [el.replace('_','[')+']' for el in df_encoded.columns]
    df = pd.concat([df,df_encoded],axis=1).drop(cat_vars,axis=1)
    del df_to_encode, df_encoded
    #print('All categorical variables are encoded')
    # Returns the dataframe with the one-hot-encoded categorical variables
    return df
# =======================================================================================================
# Categorical variable encoding
# =======================================================================================================


# =======================================================================================================
# Standard Scaling
# =======================================================================================================
def standard_scaling(df,num_vars,output_path):
    # Copies the fataframe
    df = df.copy()
    df_scale = df[num_vars]
    # Trains the Standard Scaler
    sc_X = MinMaxScaler()
    sc_X = sc_X.fit(df_scale)
    # Saves the model
    joblib.dump(sc_X, output_path)
    # Scales the variables
    df[df_scale.columns] = sc_X.transform(df_scale)
    # Returns the dataframe with the variables scaled
    #print('All variables are normalized and scaled')
    return df
# =======================================================================================================
# Standard Scaling
# =======================================================================================================


# =======================================================================================================
# Outliers detector and removal
# =======================================================================================================
def outlier_detector(df):
    # Copies the dataframe
    df = df.copy()
    
    # Computes the first and third quantiles
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    #Computes the inter-quantile range
    IQR = Q3 - Q1
    # Removes all rows that contain outliers
    df = df[~((df < (Q1-1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    # Returns a dataframe with the outliers removed
    print('Outliers removed')
    return df
# =======================================================================================================
# Outliers detector and removal
# =======================================================================================================

# -------------------------------------------------------------------------------------------------------
# =======================================================================================================
# END OF THE PREPROCESSING FOR  THE TRAINING SUBSAMPLE
# =======================================================================================================

 
    

# BEGINNING OF THE PREPROCESSING FOR  THE TEST SUBSAMPLE
# ======================================================
# -------------------------------------------------------------------------------------------------------

# =======================================================================================================
# Imputes missing values in the test subsample
# =======================================================================================================
def impute_test_data(df,imputed_path):
    # Copies the dataframe
    df = df.copy()
    # Checks for missing data
    missing = check_missing_values(df)
    if len(missing)!=0:
        # Loads only the needed imputed training data
        imputed_train = pd.read_csv(imputed_path,usecols=missing.index)
        for el in missing.index:
            if df[el].dtype.kind in 'biufc':
                # Imputes the missing values of a numerical variable with the median 
                df[el] = df[el].fillna(imputed_train[el].median())
            else:
                # Imputes the missing values of a categorial variable with the mode
                df[el] = df[el].fillna(imputed_train[el].mode()[0])
    # Returns the dataframe with the missing values imputed
    #print('Missing values imputed')
    return df
# =======================================================================================================
# Imputes missing values in the test subsample
# =======================================================================================================
   

# =======================================================================================================
# Standard scaling in the test subsample
# =======================================================================================================
def scaling_test_data(df,num_vars,scaler_path):
    # Copies the dataframe
    df = df.copy()
    df_scale = df[num_vars]
    # Loads the Standard Scaler model from path
    scaler = joblib.load(scaler_path)
    df[df_scale.columns] = scaler.transform(df_scale)
    # Return scaled dataframe
    #print('All variables are normalized and scaled')
    return df
# =======================================================================================================
# Standard scaling in the test subsample
# =======================================================================================================


# -------------------------------------------------------------------------------------------------------
# ================================================
# END OF THE PREPROCESSING FOR  THE TEST SUBSAMPLE