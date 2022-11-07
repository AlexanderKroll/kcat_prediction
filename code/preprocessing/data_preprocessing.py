from zeep import Client
import hashlib
import requests
from urllib.request import urlopen, Request
import pandas as pd
import numpy as np
from os.path import join
import os
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
import pickle

datasets_dir2 = "C:\\Users\\alexk\\substrateprediction-main\\data"

wsdl = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
password = hashlib.sha256("password".encode("utf-8")).hexdigest()
email = "alexander.kroll@hhu.de"
client = Client(wsdl)

headers= {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) ' 
                      'AppleWebKit/537.11 (KHTML, like Gecko) '
                      'Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}


parameters = (email,password)



def download_kcat_from_Brenda(EC):
    reg_url = "https://www.brenda-enzymes.org/enzyme.php?ecno=" + EC
    req = Request(url=reg_url, headers=headers) 
    html = str(urlopen(req).read())
    html = html[html.find('<a name="TURNOVER NUMBER [1/s]"></a>') : ]
    html = html[ : html.find(" </div>\\n")]
    return(html)

def add_kcat_for_EC_number(brenda_df, EC):
    html = download_kcat_from_Brenda(EC = EC)
    
    entry = 0
    sub_entry = 0
    found_entry = True
    while found_entry == True:

        data = get_entry_from_html(html = html, entry = entry, sub_entry = sub_entry)
        if data != []:
            found_entry = True
            sub_entry +=1
            brenda_df = brenda_df.append({"EC": EC, "kcat VALUE" : data[0]},
                                          ignore_index= True)
        elif sub_entry == 0:
            found_entry = False
        else:
            entry +=1
            sub_entry = 0
    return(brenda_df)

def process_string(string):
    
    if string[0] == "<" or string[1] == "<":
        string = string[string.find(">")+1:]
    string = string.replace("\\", "")
    string = string.replace("</a>", "")
    return(string)

def get_entry_from_html(html, entry, sub_entry = 0):
    ID = "tab44r" + str(entry) + "sr" + str(sub_entry) + "c"
    data = []
    for i in range(6):
        search_string = '<div id="'+ ID + str(i) + '" class="cell"><span>'
        pos = html.find(search_string) 
        if pos == -1: #string was not found, try again with different string
            search_string = '<div id="'+ ID + str(i) + '" class="cell notopborder">'
            pos = html.find(search_string)
            if pos == -1: #string was not found, try again with different string
                search_string = '<div id="'+ ID + str(i) + '" class="cell"><span'
                pos = html.find(search_string)
        if pos != -1: #string was found
            subtext = html[pos+len(search_string):]
            data.append(subtext[:subtext.find("\\n")])
        else: 
            return([])
    return(data)

def get_max_for_EC_number(EC):
    df = pd.DataFrame(columns = ["EC", "kcat VALUE"])
    df = add_kcat_for_EC_number(brenda_df = df, EC = EC)
    for ind in df.index:
        try:
            df["kcat VALUE"][ind] = float(df["kcat VALUE"][ind])
        except ValueError:
            df["kcat VALUE"][ind] = 0
    return(np.max(df["kcat VALUE"]))    


def mw_mets(metabolites):
    mw = 0
    for met in metabolites:
        if met != "":
            mol = Chem.inchi.MolFromInchi(met)
            mw = mw + Descriptors.MolWt(mol)
        
    return(mw)

def split_dataframe_enzyme(frac, df):
    df1 = pd.DataFrame(columns = list(df.columns))
    df2 = pd.DataFrame(columns = list(df.columns))
    
    #n_training_samples = int(cutoff * len(df))
    
    df.reset_index(inplace = True, drop = True)
    
    #frac = int(1/(1- cutoff))
    
    train_indices = []
    test_indices = []
    ind = 0
    while len(train_indices) +len(test_indices) < len(df):
        if ind not in train_indices and ind not in test_indices:
            if ind % frac != 0:
                n_old = len(train_indices)
                train_indices.append(ind)
                train_indices = list(set(train_indices))

                while n_old != len(train_indices):
                    n_old = len(train_indices)

                    training_seqs= list(set(df["Sequence"].loc[train_indices]))

                    train_indices = train_indices + (list(df.loc[df["Sequence"].isin(training_seqs)].index))
                    train_indices = list(set(train_indices))
                
            else:
                n_old = len(test_indices)
                test_indices.append(ind)
                test_indices = list(set(test_indices)) 

                while n_old != len(test_indices):
                    n_old = len(test_indices)

                    testing_seqs= list(set(df["Sequence"].loc[test_indices]))

                    test_indices = test_indices + (list(df.loc[df["Sequence"].isin(testing_seqs)].index))
                    test_indices = list(set(test_indices))
                
        ind +=1
    
    
    df1 = df.loc[train_indices]
    df2 = df.loc[test_indices]
    
    return(df1, df2)

def split_dataframe(frac, df):
    df1 = pd.DataFrame(columns = list(df.columns))
    df2 = pd.DataFrame(columns = list(df.columns))
    
    #n_training_samples = int(cutoff * len(df))
    
    df.reset_index(inplace = True, drop = True)
    
    #frac = int(1/(1- cutoff))
    
    train_indices = []
    test_indices = []
    ind = 0
    while len(train_indices) +len(test_indices) < len(df):
        if ind not in train_indices and ind not in test_indices:
            if ind % frac != 0:
                n_old = len(train_indices)
                train_indices.append(ind)
                train_indices = list(set(train_indices))

                while n_old != len(train_indices):
                    n_old = len(train_indices)

                    training_seqs= list(set(df["Sequence"].loc[train_indices]))
                    training_fps = list(set(df["structural_fp"].loc[train_indices]))

                    train_indices = train_indices + (list(df.loc[df["Sequence"].isin(training_seqs)].index) +
                                                           list(df.loc[df["structural_fp"].isin(training_fps)].index))
                    train_indices = list(set(train_indices))
                
            else:
                n_old = len(test_indices)
                test_indices.append(ind)
                test_indices = list(set(test_indices))

                while n_old != len(test_indices):
                    n_old = len(test_indices)

                    testing_seqs= list(set(df["Sequence"].loc[test_indices]))
                    testing_fps = list(set(df["structural_fp"].loc[test_indices]))

                    test_indices = test_indices + (list(df.loc[df["Sequence"].isin(testing_seqs)].index) +
                                                           list(df.loc[df["structural_fp"].isin(testing_fps)].index))
                    test_indices = list(set(test_indices))
                
        ind +=1
    
    
    df1 = df.loc[train_indices]
    df2 = df.loc[test_indices]
    
    return(df1, df2)

small_metabolites = ["H+", "H2O", "Zn+", "O2", "Fe+", "Mn+", "Mg+", "Na+", "Cu+"]
small_inchis = ["InChI=1S/p+1", "InChI=1S/H2O/h1H2/i/hD2", "InChI=1S/Zn", "InChI=1S/Zn/q+2",
               "InChI=1S/O2/c1-2", "InChI=1S/HO2/c1-2/h1H/p-1",
               "InChI=1S/Fe", "InChI=1S/Fe/q+3", "InChI=1S/Fe/q+2",
               "InChI=1S/Mn", "InChI=1S/Mn/q+2", "InChI=1S/Mn/q+3",
               "InChI=1S/Mg", "InChI=1S/Mg/q+2", "InChI=1S/Mg/i1+1",
               "InChI=1S/Na ", "InChI=1S/Na/q+1", "InChI=1S/Cu", 
                "InChI=1S/Cu/q+2", "InChI=1S/Cu/q+1"]

def drop_small_metabolites(metabolites):
    for met in metabolites:
        if met in small_metabolites:
            metabolites.remove(met)
    return(metabolites)



def calculate_atom_and_bond_feature_vectors(mol_files):
    #check if feature vectors have already been calculated:
    try:
            os.mkdir(join("..", "..", "data", "metabolite_data", "ts_fp_data\\mol_feature_vectors\\"))
    except FileExistsError:
        None

    #existing feature vector files:
    feature_files = os.listdir(join("..", "..", "data", "metabolite_data", "ts_fp_data\\mol_feature_vectors\\"))
    for mol_file in mol_files:
        #check if feature vectors were already calculated:
        if not mol_file + "-atoms.txt" in  feature_files:
            #load mol_file
            is_InChi = (mol_file[0:5] == "InChI")
            if is_InChi:
                mol = Chem.inchi.MolFromInchi(mol_file)
                mol_file = Inchi_dict[mol_file]
            else:
                mol = Chem.MolFromMolFile(datasets_dir2 +  "/mol-files/" + mol_file + '.mol')
            if not mol is None:
                calculate_atom_feature_vector_for_mol_file(mol, mol_file)
                calculate_bond_feature_vector_for_mol_file(mol, mol_file)
                
def calculate_atom_feature_vector_for_mol_file(mol, mol_file):
    #get number of atoms N
    N = mol.GetNumAtoms()
    atom_list = []
    for i in range(N):
        features = []
        atom = mol.GetAtomWithIdx(i)
        features.append(atom.GetAtomicNum()), features.append(atom.GetDegree()), features.append(atom.GetFormalCharge())
        features.append(str(atom.GetHybridization())), features.append(atom.GetIsAromatic()), features.append(atom.GetMass())
        features.append(atom.GetTotalNumHs()), features.append(str(atom.GetChiralTag()))
        atom_list.append(features)
    with open(join("..", "..", "data", "metabolite_data",
                   "ts_fp_data\\mol_feature_vectors\\" + mol_file + "-atoms.txt"), "wb") as fp:   #Pickling
        pickle.dump(atom_list, fp)
            
def calculate_bond_feature_vector_for_mol_file(mol, mol_file):
    N = mol.GetNumBonds()
    bond_list = []
    for i in range(N):
        features = []
        bond = mol.GetBondWithIdx(i)
        features.append(bond.GetBeginAtomIdx()), features.append(bond.GetEndAtomIdx()),
        features.append(str(bond.GetBondType())), features.append(bond.GetIsAromatic()),
        features.append(bond.IsInRing()), features.append(str(bond.GetStereo()))
        bond_list.append(features)
    with open(join("..", "..", "data", "metabolite_data",
                   "ts_fp_data\\mol_feature_vectors\\" + mol_file + "-bonds.txt"), "wb") as fp:   #Pickling
        pickle.dump(bond_list, fp)
        
        
N = 114 #maximal number of atoms in a molecule
F1 = 32         # feature dimensionality of atoms
F2 = 10         # feature dimensionality of bonds
F = F1 + F2

#Create dictionaries for the bond features:
dic_bond_type = {'AROMATIC': np.array([0,0,0,1]), 'DOUBLE': np.array([0,0,1,0]),
                 'SINGLE': np.array([0,1,0,0]), 'TRIPLE': np.array([1,0,0,0])}

dic_conjugated =  {0.0: np.array([0]), 1.0: np.array([1])}

dic_inRing = {0.0: np.array([0]), 1.0: np.array([1])}

dic_stereo = {'STEREOANY': np.array([0,0,0,1]), 'STEREOE': np.array([0,0,1,0]),
              'STEREONONE': np.array([0,1,0,0]), 'STEREOZ': np.array([1,0,0,0])}


##Create dictionaries, so the atom features can be easiliy converted into a numpy array

#all the atomic numbers with a total count of over 200 in the data set are getting their own one-hot-encoded
#vector. All the otheres are lumped to a single vector.
dic_atomic_number = {0.0: np.array([1,0,0,0,0,0,0,0,0,0]), 1.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     3.0: np.array([0,0,0,0,0,0,0,0,0,1]),  4.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     5.0: np.array([0,0,0,0,0,0,0,0,0,1]),  6.0: np.array([0,1,0,0,0,0,0,0,0,0]),
                     7.0:np.array([0,0,1,0,0,0,0,0,0,0]),  8.0: np.array([0,0,0,1,0,0,0,0,0,0]),
                     9.0: np.array([0,0,0,0,1,0,0,0,0,0]), 11.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     12.0: np.array([0,0,0,0,0,0,0,0,0,1]), 13.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     14.0: np.array([0,0,0,0,0,0,0,0,0,1]), 15.0: np.array([0,0,0,0,0,1,0,0,0,0]),
                     16.0: np.array([0,0,0,0,0,0,1,0,0,0]), 17.0: np.array([0,0,0,0,0,0,0,1,0,0]),
                     19.0: np.array([0,0,0,0,0,0,0,0,0,1]), 20.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     23.0: np.array([0,0,0,0,0,0,0,0,0,1]), 24.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     25.0: np.array([0,0,0,0,0,0,0,0,0,1]), 26.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     27.0: np.array([0,0,0,0,0,0,0,0,0,1]), 28.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     29.0: np.array([0,0,0,0,0,0,0,0,0,1]), 30.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     32.0: np.array([0,0,0,0,0,0,0,0,0,1]), 33.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     34.0: np.array([0,0,0,0,0,0,0,0,0,1]), 35.0: np.array([0,0,0,0,0,0,0,0,1,0]),
                     37.0: np.array([0,0,0,0,0,0,0,0,0,1]), 38.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     42.0: np.array([0,0,0,0,0,0,0,0,0,1]), 46.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     47.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     48.0: np.array([0,0,0,0,0,0,0,0,0,1]), 50.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     51.0: np.array([0,0,0,0,0,0,0,0,0,1]), 52.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     53.0: np.array([0,0,0,0,0,0,0,0,0,1]), 54.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     56.0: np.array([0,0,0,0,0,0,0,0,0,1]), 57.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     74.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     78.0: np.array([0,0,0,0,0,0,0,0,0,1]), 79.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     80.0: np.array([0,0,0,0,0,0,0,0,0,1]), 81.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     82.0: np.array([0,0,0,0,0,0,0,0,0,1]), 83.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     86.0: np.array([0,0,0,0,0,0,0,0,0,1]), 88.0: np.array([0,0,0,0,0,0,0,0,0,1]),
                     90.0: np.array([0,0,0,0,0,0,0,0,0,1]), 94.0: np.array([0,0,0,0,0,0,0,0,0,1])}

#There are only 5 atoms in the whole data set with 6 bonds and no atoms with 5 bonds. Therefore I lump 4, 5 and 6 bonds
#together
dic_num_bonds = {0.0: np.array([0,0,0,0,1]), 1.0: np.array([0,0,0,1,0]),
                 2.0: np.array([0,0,1,0,0]), 3.0: np.array([0,1,0,0,0]),
                 4.0: np.array([1,0,0,0,0]), 5.0: np.array([1,0,0,0,0]),
                 6.0: np.array([1,0,0,0,0])}

#Almost alle charges are -1,0 or 1. Therefore I use only positiv, negative and neutral as features:
dic_charge = {-4.0: np.array([1,0,0]), -3.0: np.array([1,0,0]),  -2.0: np.array([1,0,0]), -1.0: np.array([1,0,0]),
               0.0: np.array([0,1,0]),  1.0: np.array([0,0,1]),  2.0: np.array([0,0,1]),  3.0: np.array([0,0,1]),
               4.0: np.array([0,0,1]), 5.0: np.array([0,0,1]), 6.0: np.array([0,0,1])}

dic_hybrid = {'S': np.array([0,0,0,0,1]), 'SP': np.array([0,0,0,1,0]), 'SP2': np.array([0,0,1,0,0]),
              'SP3': np.array([0,1,0,0,0]), 'SP3D': np.array([1,0,0,0,0]), 'SP3D2': np.array([1,0,0,0,0]),
              'UNSPECIFIED': np.array([1,0,0,0,0])}

dic_aromatic = {0.0: np.array([0]), 1.0: np.array([1])}

dic_H_bonds = {0.0: np.array([0,0,0,1]), 1.0: np.array([0,0,1,0]), 2.0: np.array([0,1,0,0]),
               3.0: np.array([1,0,0,0]), 4.0: np.array([1,0,0,0]), 5.0: np.array([1,0,0,0]),
               6.0: np.array([1,0,0,0])}

dic_chirality = {'CHI_TETRAHEDRAL_CCW': np.array([1,0,0]), 'CHI_TETRAHEDRAL_CW': np.array([0,1,0]),
                 'CHI_UNSPECIFIED': np.array([0,0,1])}


def create_bond_feature_matrix(mol_name, N =114):
    '''create adjacency matrix A and bond feature matrix/tensor E'''
    try:
        with open(join("..", "..", "data", "metabolite_data",
                       "ts_fp_data", "mol_feature_vectors", mol_name + "-bonds.txt"), "rb") as fp:   # Unpickling
            bond_features = pickle.load(fp)
    except FileNotFoundError:
        return(None)
    A = np.zeros((N,N))
    E = np.zeros((N,N,10))
    for i in range(len(bond_features)):
        line = bond_features[i]
        start, end = line[0], line[1]
        A[start, end] = 1 
        A[end, start] = 1
        e_vw = np.concatenate((dic_bond_type[line[2]], dic_conjugated[line[3]],
                               dic_inRing[line[4]], dic_stereo[line[5]]))
        E[start, end, :] = e_vw
        E[end, start, :] = e_vw
    return(A,E)


def create_atom_feature_matrix(mol_name, N =114):
    try:
        with open(join("..", "..", "data", "metabolite_data",
                       "ts_fp_data", "mol_feature_vectors", mol_name + "-atoms.txt"), "rb") as fp:   # Unpickling
            atom_features = pickle.load(fp)
    except FileNotFoundError:
        print("File not found for %s" % mol_name)
        return(None)
    X = np.zeros((N,32))
    if len(atom_features) >=N:
        print("More than %s (%s) atoms in molcuele %s" % (N,len(atom_features), mol_name))
        return(None)
    for i in range(len(atom_features)):
        line = atom_features[i]
        try:
            atomic_number_mapping = dic_atomic_number[line[0]]
        except KeyError:
            atomic_number_mapping = np.array([0,0,0,0,0,0,0,0,0,1])
        x_v = np.concatenate((atomic_number_mapping, dic_num_bonds[line[1]], dic_charge[line[2]],
                             dic_hybrid[line[3]], dic_aromatic[line[4]], np.array([line[5]/100.]),
                             dic_H_bonds[line[6]], dic_chirality[line[7]]))
        X[i,:] = x_v
    return(X)


def concatenate_X_and_E(X, E, N = 114, F = 32+10):
    XE = np.zeros((N, N, F))
    for v in range(N):
        x_v = X[v,:]
        for w in range(N):
            XE[v,w, :] = np.concatenate((x_v, E[v,w,:]))
    return(XE)



def create_input_data_for_GNN_for_substrates(substrate_ID, print_error = False):
    try:
        x = create_atom_feature_matrix(mol_name = substrate_ID, N =N)
        if not x is None: 
            a,e = create_bond_feature_matrix(mol_name = substrate_ID, N =N)
            a = np.reshape(a, (N,N,1))
            xe = concatenate_X_and_E(x, e)
            return([np.array(xe), np.array(x), np.array(a)])
        else:
            if print_error:
                print("Could not create input for substrate ID %s" %substrate_ID)      
            return(None, None, None)
    except:
        print("Error for substrate ID %s" % substrate_ID)
        return(None, None, None)
    
    
def create_input_data_for_GNN_for_substrates(substrate_ID, print_error = False):
    try:
        x = create_atom_feature_matrix(mol_name = substrate_ID, N =N)
        if not x is None: 
            a,e = create_bond_feature_matrix(mol_name = substrate_ID, N =N)
            a = np.reshape(a, (N,N,1))
            xe = concatenate_X_and_E(x, e, N = 114)
            return([np.array(xe), np.array(x), np.array(a)])
        else:
            if print_error:
                print("Could not create input for substrate ID %s" %substrate_ID)      
            return(None, None, None)
    except:
        return(None, None, None)
