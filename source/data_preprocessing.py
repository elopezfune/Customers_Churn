# Data prepocessing module
# ========================

# Standard libraries
# ==================
import joblib, os, re, datetime
import numpy as np
from numpy import array
import pandas as pd
from string import digits

# Preprocessing libraries
# =====================================
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Machine and Deep Learning libraries
# ===================================
# Kaplan-Meier model
from pysurvival.models.non_parametric import KaplanMeierModel

## Time series
#from sklearn.ensemble import RandomForestRegressor

# Survival model
from pysurvival.models.survival_forest import ConditionalSurvivalForestModel
from pysurvival.utils import save_model

# Metrics
# =======
#from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from pysurvival.utils.metrics import concordance_index


# Optimizer
# =========
from scipy.optimize import basinhopping

# Neglect warnings
# ================
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



# Reduces memory usage in data
# ===========================
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


# Checks for duplicated data and removes them
# ===========================================
def duplicated_data(df):
    # Copies the dataframe
    df = df.copy()
    
    # Rows containing duplicate data
    print("Removed ", df[df.duplicated()].shape[0], ' duplicate rows')
    
    # Returns a dataframe with the duplicated rows removed
    return df.drop_duplicates()


# Missing values
# ==============
# Checks for columns with missing values (NaNs)
# =============================================
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

# Missing values imputation
# =========================
def data_imputer(df,target):
    # Copies the dataframe
    df = df.copy()
    
    missing = check_missing_values(df)
    if len(missing)!=0:
        for el in missing.index:
            if df[el].dtype.kind in 'biufc':
                # Imputes the missing values of a numerical variable with the median
                df[el] = df[[el,target]].groupby(str(target)).transform(lambda x: x.fillna(x.median()))
            else:
                # Imputes the missing values of a categorial or string variable with the mode
                df[el] = df[[el,target]].groupby(str(target)).transform(lambda x: x.fillna(x.mode()[0]))
    print('Missing values imputed')
    # Returns a dataframe with the missing values imputed
    return df



# Categorical variable encoding
# =============================
def categorical_encoding(df):
    # Copies the fataframe
    df = df.copy()
    # Returns the dataframe with the one-hot-encoded categorical variables
    print('All categorical variables are encoded')
    return pd.get_dummies(df)



# Standard Scaling
# ================
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
    print('All variables are normalized and scaled')
    return df


# Outliers detector
# =================
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
    print('Outliers removed.')
    return df



# Models
# ======

# Kaplan-Meier survival table
# ===========================
def kaplan_meier_survival(T,E,CI,start_date):
    # Initializing the KaplanMeierModel
    model = KaplanMeierModel()
    # Fitting the model and retrieving the 95% confidence intervals
    model.fit(T, E, alpha=CI)
    # Displaying the survival data
    survival_dataframe = model.survival_table
    survival_dataframe['Number of events'] = survival_dataframe['Number of events'].astype(int)
    survival_dataframe['Number at risk'] = survival_dataframe['Number at risk'].astype(int)
    # Creates the date feature
    survival_dataframe['Date'] = pd.to_datetime(start_date)+survival_dataframe['Time'].apply(lambda x: pd.offsets.DateOffset(months=x))
    # Returns the survival dataframe
    return survival_dataframe


# Time Series
# ===========

## Split a univariate sequence into samples
## ========================================    
#def split_sequence(sequence, n_steps):
#    X, y = list(), list()
#    for i in range(len(sequence)):
#        # Find the end of this pattern
#        end_ix = i + n_steps
#        # Check if we are beyond the sequence
#        if end_ix > len(sequence)-1:
#            break
#        # Gather input and output parts of the pattern
#        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
#        X.append(seq_x)
#        y.append(seq_y)
#    return array(X), array(y)


#def random_forest_regresor(df,date,target,partition,output_path):
#    # Copies the dataframe
#    df = df.copy()
#    # Filters out the time series
#    df = df[[date,target]]
#    # Define input sequence
#    raw_seq = df[target]
#    # Choose a number of time steps
#    n_steps = partition
#    # Split into samples
#    X, y = split_sequence(raw_seq, n_steps)
#
#    # Defines the model
#    def random_forest_model(X_train,y_train,params):
#        n_estimators, random_state, ccp_alpha = params
#        model = RandomForestRegressor(n_estimators=int(n_estimators), 
#                                      n_jobs=-1,
#                                      random_state=int(random_state),
#                                      ccp_alpha=ccp_alpha)
#        # Fits the model
#        model = model.fit(X_train,y_train)
#        # Returns the fitted model
#        return model
#    
#    # Defines the metric to optimize the hyper parameters of the model
#    def metric(params):
#        # Creates train+validation subsamples
#        X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3)
#        # Loads the model and make predictions
#        model = random_forest_model(X_train,y_train,params)
#        y_pred = model.predict(X_valid)
#        # Returns the root mean square error, which will be minimized later
#        return mean_squared_error(y_valid, y_pred)
#    
#    # This is the optimizer of the model
#    # Hyperparameters: n_estimators, random_state, ccp_alpha
#    boundary = [(10.0, 300.0), (0, 100), (0.0, 1.0)]
#    # Initial guess point
#    x0 = [np.mean(el) for el in boundary]
#    
#    #Minimizer using the Basin-Hoping algorithm
#    minimizer_kwargs = dict(method="L-BFGS-B", bounds=boundary)
#    opt = basinhopping(metric, x0, minimizer_kwargs=minimizer_kwargs)
#    
#    # Prints the concordance index for the internal validation cohort
#    print("The validation metric is: ", opt.fun)
#    print("  ")
#    print("The optimized hyper-parameters are: ")
#    print("n_estimators is: ", int(opt.x[0]))
#    print("random_state is: ", int(opt.x[1]))
#    print("ccp_alpha is: ", round(opt.x[2],2))
#    
#    
#    #Defines the optimized model already fitted
#    model = random_forest_model(X,y,opt.x)
#    
#    # Saves the model
#    joblib.dump(model,output_path)
#    
#    # Returns the fitted model
#    return model

## Time Series predictor
#def time_series_predictor(df,date,target,model_path,forecast):
#    # Copies the dataframe
#    df = df.copy()
#    df = df[[date,target]]
#    # Loads the model
#    model = joblib.load(model_path)
#    # Reads the number of time steps from the trained model
#    n_steps = model.n_features_in_
#    # Split into samples
#    raw_seq = df[target]
#    X, y = split_sequence(raw_seq, n_steps)
#    
#    # Creates prediction table
#    y_pred = model.predict(X)
#    if raw_seq.dtype=="int64":
#        y_pred = y_pred.astype(int)
#    nans = [np.NaN]*(n_steps)
#    df[target+' (Prediction)'] = nans+list(y_pred)
#    
#    # Forecasting the next values
#    X_for = [X[-1]]
#    y_for = []
#    for i in range(int(forecast)):
#        result = model.predict(X_for)
#        y_for.append(int(result[0]))
#        X_for = [np.delete(np.append(X_for,int(result)), 0)]
#        
#    # Returns the dataframe
#    return df



# Multi-class classifier
# ======================
def conditional_survival_forest(df, timeline, target, output_path):
    # Copies the dataframe
    df = df.copy()
    
    # Trains the model
    def train_model(X_train, T_train, E_train, params):
        num_trees, sample_size_pct, seed = params
        # Defines the model
        model = ConditionalSurvivalForestModel(num_trees = int(num_trees))
        model = model.fit(X_train, T_train, E_train,
                          sample_size_pct = sample_size_pct,
                          importance_mode = 'impurity_corrected',
                          seed = int(seed),
                          save_memory=True)
        return model

    # Defines the metric to optimize the hyper parameters of the model
    def metric(params):
        # Splits into train and internal validation subsamples
        X_train, X_valid = train_test_split(df,test_size=0.3)
        T_train, T_valid = X_train[timeline], X_valid[timeline]
        E_train, E_valid = X_train[target], X_valid[target]
        X_train, X_valid = X_train.drop([timeline,target],axis=1), X_valid.drop([timeline,target],axis=1)
        
        # Loads and train the model with the training subsample
        model = train_model(X_train, T_train, E_train, params)
        # Computes the metric with the validation subsample
        c_index = concordance_index(model, X_valid, T_valid, E_valid)
        # Returns 1-c index to be minimized
        return 1.0-c_index

    # This is the optimizer of the model
    # n_estimators, learning_rate, random_state, 
    boundary = [(10, 300), (0.1, 0.9), (0, 100)]
    # Initial point, not needed for the differential_evolution minimizer
    x0 = [np.mean(el) for el in boundary]
    
    # Minimizer using the Differential evolution algorithm
    #opt = differential_evolution(metric, bounds=boundary,seed=1)
    
    # Minimizer using the Basin-Hoping algorithm
    # Uses the method L-BFGS-B because the problem is smooth and bounded
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=boundary)
    opt = basinhopping(metric, x0, minimizer_kwargs=minimizer_kwargs)
    
    # Prints the validation metric and the optimal parameters
    print("The Concordance Index is: ", 1.0-opt.fun)
    print("  ")
    print("The optimized hyper-parameters are: ")
    print("num_trees is: ", int(opt.x[0]))
    print("sample_size_pct is: ", opt.x[1])
    print("seed is: ", int(opt.x[2]))
    
    # Defines the optimized model with the whole training data
    model = train_model(df.drop([timeline,target],axis=1), df[timeline], df[target], opt.x)
    # Saves the model
    save_model(model, output_path)
    print('Finished training and model production.')
    return model
 
    
    
    
    
    

# Make predictions
# ================

    
    
# Individual pre-processing for the test subsample
# ================================================

# Imputes missing data in the test subsample
# ==========================================
def impute_test_data(df,output_path):
    # Copies the dataframe
    df = df.copy()
    # Checks for missing data
    missing = check_missing_values(df)
    if len(missing)!=0:
        # Loads the imputed training data
        imputed_train = pd.read_csv(output_path)
        for el in missing.index:
            if df[el].dtype == float:
                # Imputes the missing values of a numerical variable with the median
                df[el] = df[el].fillna(imputed_train[el].median())
            else:
                # Imputes the missing values of a categorial or string variable with the mode
                df[el] = df[el].fillna(imputed_train[el].mode()[0])
    # Returns the dataframe with the missing values imputed
    print('Missing values imputed.')
    return df
