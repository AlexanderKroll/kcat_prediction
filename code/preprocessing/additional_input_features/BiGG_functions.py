import pandas as pd
import os
from os.path import join
import numpy as np
import json
import requests


def array_column_to_strings(df, column):
    df[column] = [str(list(df[column][ind])) for ind in df.index]
    return(df)

def string_column_to_array(df, column):
    df[column] = [np.array(eval(df[column][ind])) for ind in df.index]
    return(df)

def adding_KEGG_CIDS_to_model_df(model_df, model_name):
    with open(join("..", "..", "..", "data", "BiGG_data",  model_name + ".json")) as json_file:
        model = json.load(json_file)
    model = model["metabolites"]

    met_IDs = []
    for i in range(len(model)):
        met_IDs.append(model[i]["id"])

    model_df["substrate KEGG CIDs"] = np.nan
    model_df["product KEGG CIDs"] = np.nan
    
    for ind in model_df.index:
        metabolites = model_df["substrates"][ind]
        CIDs = []
        for met in metabolites:
            try:
                CIDs = CIDs  + model[met_IDs.index(met)]["annotation"]["kegg.compound"]
            except KeyError:
                None
        if CIDs != []:
            model_df["substrate KEGG CIDs"][ind] = CIDs
        else:
            model_df["substrate KEGG CIDs"][ind] = np.nan
            
        metabolites = model_df["products"][ind]
        CIDs = []
        for met in metabolites:
            try:
                CIDs = CIDs  + model[met_IDs.index(met)]["annotation"]["kegg.compound"]
            except KeyError:
                None
        if CIDs != []:
            model_df["product KEGG CIDs"][ind] = CIDs
        else:
            model_df["product KEGG CIDs"][ind] = np.nan
            
        
    return(model_df)


def create_metabolic_model_df(model_path, model_name):
    #load model as json-file:
    try:
        with open(model_path) as json_file:
            model = json.load(json_file)
    except: # download it if not already done:
        r = requests.get("http://bigg.ucsd.edu/api/v2/models/" + model_name+ "/download")
        with open(join(model_path, model_name +".json"), 'w') as outfile:
            json.dump(r.json(), outfile)
        model = r.json()
    
    model = model["reactions"]

    model_df = pd.DataFrame(columns = ["BiGG ID", "substrates", "products"])

    for i in range(len(model)):
        substrates = []
        products = []
        dict_met = model[i]["metabolites"]
        for key in dict_met.keys():
            if dict_met[key] < 0:
                substrates.append(key)
            else:
                products.append(key)
        substrates.sort(), products.sort()
       
        model_df = model_df.append({"BiGG ID" : model[i]["id"],
                                       "substrates" : substrates,
                                       "products" : products},
                                       ignore_index = True)
      
    return(model_df)

def find_kegg_id(annotation_list):
    for annotation in annotation_list:
        if annotation[0] == 'KEGG Compound':
            return(annotation[1].split("/")[-1])
    return(np.nan)

def find_chebi_id(annotation_list):
    for annotation in annotation_list:
        if annotation[0] == 'CHEBI':
            return(annotation[1].split("/")[-1])
    return(np.nan)

def find_mnx_id(annotation_list):
    for annotation in annotation_list:
        if annotation[0] == 'MetaNetX (MNX) Chemical':
            return(annotation[1].split("/")[-1])
    return(np.nan)

def find_inchi_key(annotation_list):
    for annotation in annotation_list:
        if annotation[0] == 'InChI Key':
            return(annotation[1].split("/")[-1])
    return(np.nan)

def adding_CIDS_to_model_df(model_df, metabolites_df):


    model_df["substrate CIDs"] = ""
    model_df["product CIDs"] = ""
    model_df["complete"] = np.nan
    
    for ind in model_df.index:
        complete = False
        sub_ID_list, complete_subs = get_ID_list(metabolites = model_df["substrates"][ind], metabolites_df = metabolites_df)
        pro_ID_list, complete_pros = get_ID_list(metabolites = model_df["products"][ind], metabolites_df = metabolites_df)
        
        if complete_subs and complete_pros:
            complete = True
        
        model_df["substrate CIDs"][ind] = sub_ID_list
        model_df["product CIDs"][ind] = pro_ID_list
        model_df["complete"][ind] = complete            
        
    return(model_df)


def get_ID_list(metabolites, metabolites_df):
    ID_list = []
    complete = True
    for met in metabolites:
        [kegg_id, mnx_id] = [metabolites_df.loc[met]["KEGG ID"], metabolites_df.loc[met]["MNX ID"]]
        if pd.isnull(kegg_id):
            if pd.isnull(mnx_id):
                ID_list.append(np.nan)
                complete = False
            else:
                ID_list.append(mnx_id)
        else:
            ID_list.append(kegg_id)
    return(ID_list, complete)