import json
import pandas as pd


# Loads files provided their path
# ===============================
def load_data(path,index):
    # Loads the data
    with open(path) as f:
        g = json.load(f)
    # Converts json dataset from dictionary to dataframe
    #print('Data loaded correctly')
    df = pd.DataFrame.from_dict(g)
    df = df.set_index(index)
    return df
