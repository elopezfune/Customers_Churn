# Machine Learning ToolBox
# ========================

# Standard libraries
# ==================
import joblib
import numpy as np
from numpy import array
import pandas as pd


# Preprocessing libraries
# =======================
from sklearn.model_selection import train_test_split

# Machine Learning and Survival libraries
# =======================================
# Time series
from sklearn.ensemble import RandomForestRegressor
# Survival library
from pysurvival.models.non_parametric import KaplanMeierModel
from pysurvival.models.survival_forest import ConditionalSurvivalForestModel
from pysurvival.utils import save_model
# Metrics
from sklearn.metrics import mean_squared_error
from pysurvival.utils.metrics import concordance_index

# Optimizers
# ==========
from scipy.optimize import basinhopping, differential_evolution


# Neglect warnings
# ================
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



# Machine Learning and Survival Models
# ====================================

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
# Split a univariate sequence into samples
# ========================================    
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def random_forest_regresor(df,date,target,partition,output_path):
    # Copies the dataframe
    df = df.copy()
    # Filters out the time series
    df = df[[date,target]]
    # Define input sequence
    raw_seq = df[target]
    # Choose a number of time steps
    n_steps = partition
    # Split into samples
    X, y = split_sequence(raw_seq, n_steps)

    # Defines the model
    def random_forest_model(X_train,y_train,params):
        n_estimators, random_state, ccp_alpha = params
        model = RandomForestRegressor(n_estimators=int(n_estimators), 
                                      n_jobs=-1,
                                      random_state=int(random_state),
                                      ccp_alpha=ccp_alpha)
        # Fits the model
        model = model.fit(X_train,y_train)
        # Returns the fitted model
        return model
    
    # Defines the metric to optimize the hyper parameters of the model
    def metric(params):
        # Creates train+validation subsamples
        X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3)
        # Loads the model and make predictions
        model = random_forest_model(X_train,y_train,params)
        y_pred = model.predict(X_valid)
        # Returns the root mean square error, which will be minimized later
        return mean_squared_error(y_valid, y_pred)
    
    # This is the optimizer of the model
    # Hyperparameters: n_estimators, random_state, ccp_alpha
    boundary = [(10.0, 300.0), (0, 100), (0.0, 1.0)]
    
    # Minimizer using the Differential evolution algorithm
    opt = differential_evolution(metric, bounds=boundary,seed=1)
    
    # Prints the concordance index for the internal validation cohort
    #print("The validation metric is: ", opt.fun)
    #print("  ")
    #print("The optimized hyper-parameters are: ")
    #print("n_estimators is: ", int(opt.x[0]))
    #print("random_state is: ", int(opt.x[1]))
    #print("ccp_alpha is: ", round(opt.x[2],2))
    
    
    #Defines the optimized model already fitted
    model = random_forest_model(X,y,opt.x)
    
    # Saves the model
    joblib.dump(model,output_path)
    
    # Returns the fitted model
    return model

# Time Series predictor
def time_series_predictor(df,date,target,model_path,forecast):
    # Copies the dataframe
    df = df.copy()
    df = df[[date,target]]
    # Loads the model
    model = joblib.load(model_path)
    # Reads the number of time steps from the trained model
    n_steps = model.n_features_in_
    # Split into samples
    raw_seq = df[target]
    X, y = split_sequence(raw_seq, n_steps)
    
    # Creates prediction table
    y_pred = model.predict(X)
    if raw_seq.dtype=="int64":
        y_pred = y_pred.astype(int)
    nans = [np.NaN]*(n_steps)
    df[target+' (Prediction)'] = nans+list(y_pred)
    
    # Forecasting the next values
    X_for = [X[-1]]
    y_for = []
    for i in range(int(forecast)):
        result = model.predict(X_for)
        y_for.append(int(result[0]))
        X_for = [np.delete(np.append(X_for,int(result)), 0)]
        
    # Returns the dataframe
    return df



# Survival model
# ==============
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
    # num_trees, sample_size_pct, seed, 
    boundary = [(10, 300), (0.1, 0.9), (0, 100)]
    # Initial point, not needed for the differential_evolution minimizer
    x0 = [np.mean(el) for el in boundary]
    
    # Minimizer using the Basin-Hoping algorithm
    # Uses the method L-BFGS-B because the problem is smooth and bounded
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=boundary)
    opt = basinhopping(metric, x0, minimizer_kwargs=minimizer_kwargs)
    
    # Prints the validation metric and the optimal parameters
    #print("The optimized concordance index is: ", 1.0-opt.fun)
    #print("The optimized hyper-parameters are: ")
    #print("num_trees is: ", int(opt.x[0]))
    #print("sample_size_pct is: ", opt.x[1])
    #print("seed is: ", int(opt.x[2]))
    
    # Defines the optimized model with the whole training data
    model = train_model(df.drop([timeline,target],axis=1), df[timeline], df[target], opt.x)
    # Saves the model
    save_model(model, output_path)
    #print('Finished training and model production')
    return model