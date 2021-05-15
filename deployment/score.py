#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
import json
import time
from azureml.core.model import Model
# from azureml.core.run import Run
import argparse
from scipy.stats import anderson
from imblearn.over_sampling import SMOTE
# import json
#import traceback
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
# from lightgbm import LGBMClassifier, plot_importance
# from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, precision_score, recall_score

import pandas as pd
import numpy as np

def init():
    global d['rf_model_Credit Card']
    global d['rf_model_Funds']
    global d['rf_model_Loans']
    global d['rf_model_Mortgage']
    global d['rf_model_Pensions']

    list_of_modelnames=['rf_model_Credit Card','rf_model_Funds','rf_model_Loans','rf_model_Mortgage','rf_model_Pensions']
    counts=[1,2,3,4,5]
    d={}
    for model_rname,count in zip(list_of_modelnames,counts):
        print ("model initialized" + time.strftime("%H:%M:%S"))
        model_path = Model.get_model_path(model_name = model_rname)
        d[model_rname]=load(model_path)
                
            

def dataprocessoer(val_Data, ordinals):
    
    global column_names
    
    # Column names
    column_names = ['Partition column','Customer code', 'Employment index', 'Country Residence', 'Gender', 'Age', 'Join date', 'New customer Index',
'Customer seniority', 'Primary cusotmer index', 'Last date as Primary customer', 'Customer type', 'Customer relation',
'Residence index', 'Foreigner index', 'Spouse emp index', 'Channel Index', 'Deceased index', 'Address type', 'Province code', 
'Province name', 'Activity index','Gross Income', 'Segmentation']
    
    # Independent Columns
    val_Data = val_Data.iloc[:,0:24]
    
    # Setting column names
    val_Data.columns = column_names
    
    val_Data.dropna(subset= ['Customer code'], inplace=True)
    
    # Checking for invalid customer
    if((val_Data.shape[0] == 1) and ((val_Data['Customer code'].iloc[0]=='') or (val_Data['Customer code'].isnull().any()))):
        print("Invalid Customer ID")
    else:
        pass
    
    
    val_Data.drop_duplicates(subset= ['Customer code'], inplace = True)
    
    users_code = val_Data[['Customer code']]
    
    # cus rel age
    val_Data['Join date'] = pd.to_datetime(val_Data['Join date'])
    
    x = []
    for i in val_Data['Join date']:
        if(i is pd.NaT):
            x.append(0)
        else:
            x.append(int(pd.Timedelta(pd.to_datetime('today')-i).days/365))
            
    val_Data['cust_rel_age'] = x
    
    
    
    val_Data.drop(dropbales, axis =1, inplace = True)
    val_Data.drop(['Join date','Country Residence','Partition column','Customer code',
                   'Province code','Address type'], axis =1, inplace = True)
    
    # AGE
    val_Data['Age'].isnull().sum() # Tricky
    val_Data['Age'][val_Data['Age'] == ' NA'] # There it is
    not_null_age = val_Data['Age'][val_Data['Age'] != ' NA'].astype(int)
    
    median_Age = int(not_null_age.median())
    val_Data['Age'].replace(' NA', str(median_Age), inplace= True)
    val_Data['Age'] = val_Data['Age'].astype(int)
    
    
    #CUS AGE
    val_Data['Customer seniority'] = val_Data['Customer seniority'].astype(str)
    val_Data['Customer seniority'] = val_Data['Customer seniority'].str.strip()
    val_Data['Customer seniority'].replace('-999999','0', inplace = True)
    
    customer_seniority_in_months = val_Data['Customer seniority'][(val_Data['Customer seniority']!= 'NA') & 
                                                                  (val_Data['Customer seniority'].notnull())].astype(int)
    median_sen = int(customer_seniority_in_months.median())
    val_Data['Customer seniority'].replace('NA', str(median_sen), inplace= True)
    val_Data['Customer seniority'].fillna(str(median_sen), inplace= True)
    val_Data['Customer seniority'] = val_Data['Customer seniority'].astype(int)
    
    
    # Cust type
    val_Data['Customer type'] = val_Data['Customer type'].astype(str)
    val_Data['Customer type'] = val_Data['Customer type'].str.strip()
    mode_of_indrel_1mes = val_Data['Customer type'].mode()[0]
    val_Data['Customer type'].replace('nan',mode_of_indrel_1mes, inplace = True)
    
    
    # Type casting indrel = says if a customer is primary customer throughout or not
    val_Data['Primary cusotmer index'] = val_Data['Primary cusotmer index'].astype(str)
    # Says new customer or not 1 says yes 
    val_Data['New customer Index'] = val_Data['New customer Index'].astype(str)
    # Says if the customer is active or not
    val_Data['Activity index'] =  val_Data['Activity index'].astype(str)
    
    val_Data['Primary cusotmer index'].replace('nan', '1.0', inplace= True)
    val_Data['New customer Index'].replace('nan','0.0', inplace= True)
    val_Data['Activity index'].replace('nan','0.0', inplace= True)
    
    
    #Missing value imputation
    
    numerical_null_cols = val_Data.select_dtypes(include = [np.number]).columns
    obj_null_cols = val_Data.select_dtypes(include = [np.object]).columns
    
    one_point_mods = {'Employment index': 'N',
                         'Country Residence': 'ES',
                         'Gender': 'V',
                         'New customer Index': '0.0',
                         'Primary cusotmer index': '1.0',
                         'Customer type': '1',
                         'Customer relation': 'I',
                         'Residence index': 'S',
                         'Foreigner index': 'N',
                         'Channel Index': 'KHE',
                         'Deceased index': 'N',
                         'Province name': 'MADRID',
                         'Activity index': '0.0',
                         'Segmentation': '02 - PARTICULARES'}
    
    one_point_meads = {'Customer code': 931405.5,
                         'Age': 39.0,
                         'Customer seniority': 54.0,
                         'Gross Income': 101413.54499999998}

    if(val_Data.shape[0] == 1):
        for i in numerical_null_cols:
            if(val_Data[i].isnull().iloc[0]):
                val_Data[i].fillna(one_point_meads[i], inplace= True)
            else:
                pass     
        for i in obj_null_cols:
            if(val_Data[i].isnull().iloc[0]):
                val_Data[i].fillna(one_point_mods[i], inplace= True)
            else:
                pass    
    else:
        numerical_imputer(numerical_null_cols, val_Data)
        objects_imputer(obj_null_cols, val_Data)

    for i in ordinals:
        val_Data[i] = val_Data[i].astype('category')
        val_Data[i] = val_Data[i].cat.codes
        
        
        
    return users_code,val_Data

def product_prompter(df):
    prods_prompt={}
    for i in df.index:
        r_dict = df.loc[i].sort_values(ascending = False).to_dict()  
        k = 1
        for j in r_dict:
            r_dict[j] = k
            k+=1
        prods_prompt[i] = r_dict
    return prods_prompt
    
def run(raw_data):
    try:
        column_names = ['Partition column','Customer code', 'Employment index', 'Country Residence', 'Gender', 'Age', 'Join date', 'New customer Index',
        'Customer seniority', 'Primary cusotmer index', 'Last date as Primary customer', 'Customer type', 'Customer relation',
        'Residence index', 'Foreigner index', 'Spouse emp index', 'Channel Index', 'Deceased index', 'Address type', 'Province code', 
        'Province name', 'Activity index','Gross Income', 'Segmentation']
        data = json.loads(raw_data)["data"]
        data = numpy.array(data)
        data=pd.DataFrame(data,columns=column_names)
        user_code, val_Data = dataprocessoer(data, ordinals)
        validation_set = val_Data.copy()
        list_of_modelnames=['rf_model_Credit Card','rf_model_Funds','rf_model_Loans','rf_model_Mortgage','rf_model_Pensions']

        for i in list_of_modelnames:
#             print(i)
#             rf.fit(smoted_X[i], smoted_y[i])
            # pickle them 
            probs = np.array(np.round(d[i].predict_proba(val_Data)[:,1],3), dtype = float)
            validation_set[i] = probs
        prods_recom = product_prompter(validation_set.iloc[:,-5:])
        for i in prods_recom:
            product_recomendation[user_code.loc[i][0]] = prods_recom[i]
    
#         return product_recomendation
#         result = model.predict(data)
        return json.dumps({"result": product_recomendation})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})

