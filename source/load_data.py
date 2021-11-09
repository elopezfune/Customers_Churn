import re, os, json, uuid, csv
import pandas as pd
import numpy as np
import config
import pickle 

# Loads files provided their path
# ===============================
def load_data(path,index):
    # Loads the data
    with open(path) as f:
        g = json.load(f)
    # Converts json dataset from dictionary to dataframe
    print('Data loaded correctly.')
    df = pd.DataFrame.from_dict(g)
    df = df.set_index(index)
    return df
    


# Save csv file
# =============
def save_csv(df,output_path):
    # Copies the dataframe
    df = df.copy()
    # Saves the dataframe as a csv file
    
    with open(output_path, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(df)

    
#    with open(output_path,'wb') as csvFile:
#        writer = csv.writer(csvFile)
#        writer.writerows(df)
    
    
    
    
    
    
# Data models
# ===========
class Transaction_Model:
    def __init__(self):
        self.categorization_model = {}
        
    def populate_transaction(self,user_id,value,issued,credited,entity,transaction_type,macrocategory,category,account):
        self.categorization_model['user_id'] = str(user_id)+"--"+str(uuid.uuid4())   #Positive integer field + uuid-code
        self.categorization_model['value'] = value                                   #Floating number
        self.categorization_model['issued'] = issued                                 #Timestamp
        self.categorization_model['credited'] = credited                             #Timestamp
        self.categorization_model['entity'] = str(entity) +"--" +str(uuid.uuid4())   #String + uuid-code
        self.categorization_model['transaction_type'] = transaction_type             #String
        self.categorization_model['macrocategory'] = macrocategory                   #String
        self.categorization_model['category'] = category                             #String
        self.categorization_model['account'] = str(account)+"--"+str(uuid.uuid4())   #Foreign key + uuid-code
        return self.categorization_model

    
class Category_Model:
    def __init__(self):
        self.categorization_model = {}
    def populate_multiclass(self, id, value, issued, credited, entity, transaction_type, macrocategory, account, predicted_class, predicted_proba, model_ref):
        self.categorization_model['id'] = str(id)                                    #Positive integer field + uuid-code
        self.categorization_model['value'] = value                                   #Floating number
        self.categorization_model['issued'] = issued                                 #Timestamp
        self.categorization_model['credited'] = credited                             #Timestamp
        self.categorization_model['entity'] = str(entity)                            #String + uuid-code
        self.categorization_model['transaction_type'] = transaction_type             #String
        self.categorization_model['macrocategory'] = macrocategory                   #String
        self.categorization_model['account'] = str(account)                          #Foreign key + uuid-code
        self.categorization_model['predicted_class'] = predicted_class               #String
        self.categorization_model['Probability'] = predicted_proba                   #Floating number
        self.categorization_model["model_ref"] = model_ref                           #String
        return self.categorization_model


# Loads pickle models
# ============
def pickle_loader(path):
    pkl_file = open(path, 'rb')
    model = pickle.load(pkl_file)
    pkl_file.close()
    return model
        

def save_data(path, name, data):
    return data.to_csv(path+'/'+str(name))