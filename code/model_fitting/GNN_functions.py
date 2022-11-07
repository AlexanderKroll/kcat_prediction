import os
from os.path import join
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import random
import torch.optim as optim
from sklearn.metrics import r2_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CURRENT_DIR = join("..", "..", "data", "metabolite_data")


def load_data_train(ind, split):           
    XE_sub = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN_hyperopt2", str(ind)+"_XE_sub.npy"))
    XE_pro = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN_hyperopt2", str(ind)+"_XE_pro.npy"))
    X_sub = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN_hyperopt2", str(ind)+"_X_sub.npy"))
    X_pro = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN_hyperopt2", str(ind)+"_X_pro.npy"))
    A_sub = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN_hyperopt2", str(ind)+"_A_sub.npy"))
    A_pro = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN_hyperopt2", str(ind)+"_A_pro.npy"))
    structure = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN_hyperopt2", str(ind)+"_str.npy"))
    y = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN_hyperopt2", str(ind)+"_y.npy"))
    return(XE_sub, XE_pro, X_sub, X_pro, A_sub, A_pro, structure, y)


def load_batch_train(batch, split):
    XE_subs, XE_pros, X_subs, X_pros, A_subs, A_pros, structures, ys = [], [], [], [], [], [], [], []
    for ind in batch:
        XE_sub, XE_pro, X_sub, X_pro, A_sub, A_pro, structure, y = load_data(ind = ind, split = split)
        XE_subs.append(XE_sub), XE_pros.append(XE_pro), X_subs.append(X_sub), X_pros.append(X_pro)
        A_subs.append(A_sub), A_pros.append(A_pro), structures.append(structure), ys.append(y)
        print(ind, structure)
                       
    XE_subs = torch.tensor(np.array(XE_subs), dtype=torch.float32)
    XE_pros = torch.tensor(np.array(XE_pros), dtype=torch.float32)
    X_subs = torch.tensor(np.array(X_subs), dtype=torch.float32)
    X_pros = torch.tensor(np.array(X_pros), dtype=torch.float32)
    A_subs = torch.tensor(np.array(A_subs), dtype=torch.float32)
    A_pros = torch.tensor(np.array(A_pros), dtype=torch.float32)
    ys = torch.tensor(np.array(ys), dtype=torch.float32)
    
    return(XE_subs, XE_pros, X_subs, X_pros, A_subs, A_pros, structures, ys)


def load_data(ind, split):           
    XE_sub = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN", str(ind)+"_XE_sub.npy"))
    XE_pro = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN", str(ind)+"_XE_pro.npy"))
    X_sub = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN", str(ind)+"_X_sub.npy"))
    X_pro = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN", str(ind)+"_X_pro.npy"))
    A_sub = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN", str(ind)+"_A_sub.npy"))
    A_pro = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN", str(ind)+"_A_pro.npy"))
    structure = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN", str(ind)+"_str.npy"))
    y = np.load(join(CURRENT_DIR, "ts_fp_data", split + "_GNN", str(ind)+"_y.npy"))
    return(XE_sub, XE_pro, X_sub, X_pro, A_sub, A_pro, structure, y)


def load_batch(batch, split):
    XE_subs, XE_pros, X_subs, X_pros, A_subs, A_pros, structures, ys = [], [], [], [], [], [], [], []
    for ind in batch:
        XE_sub, XE_pro, X_sub, X_pro, A_sub, A_pro, structure, y = load_data(ind = ind, split = split)
        XE_subs.append(XE_sub), XE_pros.append(XE_pro), X_subs.append(X_sub), X_pros.append(X_pro)
        A_subs.append(A_sub), A_pros.append(A_pro), structures.append(structure), ys.append(y)

        
    XE_subs = torch.tensor(np.array(XE_subs), dtype=torch.float32)
    XE_pros = torch.tensor(np.array(XE_pros), dtype=torch.float32)
    X_subs = torch.tensor(np.array(X_subs), dtype=torch.float32)
    X_pros = torch.tensor(np.array(X_pros), dtype=torch.float32)
    A_subs = torch.tensor(np.array(A_subs), dtype=torch.float32)
    A_pros = torch.tensor(np.array(A_pros), dtype=torch.float32)
    ys = torch.tensor(np.array(ys), dtype=torch.float32)
    
    return(XE_subs, XE_pros, X_subs, X_pros, A_subs, A_pros, structures, ys)


def get_reaction_representations(batches, model, df, split = "train"):
    df["reaction_rep"] = ""
    model.eval()
    for batch in batches:
        XE_subs, XE_pros, X_subs, X_pros, A_subs, A_pros, structures, ys = load_batch(batch, split = split)
        XE_subs, XE_pros, X_subs, X_pros, A_subs, A_pros = XE_subs.to(device), XE_pros.to(device), X_subs.to(device), X_pros.to(device), A_subs.to(device), A_pros.to(device)
        structure_array = str(structures[0]).split("_")
        representations = model.get_rep(XE_subs, XE_pros, X_subs, X_pros, A_subs, A_pros, structure_array).cpu().detach().numpy()
        for i, ind in enumerate(batch):
            df["reaction_rep"][ind] = list(representations[i, :])
    return(df)


# Model parameters
N = 114        # maximum number of nodes
F1 = 32         # feature dimensionality of atoms
F2 = 10         # feature dimensionality of bonds
F = F1+F2




class GNN(nn.Module):
    def __init__(self, D= 100, N = 114, F1 = 32 , F2 = 10, F = F1+F2, droprate = 0.2):
        print("droprate", droprate)
        super(GNN, self).__init__()
        #first head
        self.Wi_sub = nn.Parameter(torch.empty((1,1,F,D), requires_grad = True).to(device))
        self.Wm1_sub = nn.Parameter(torch.empty((1,1,D,D), requires_grad = True).to(device)) 
        self.Wm2_sub= nn.Parameter(torch.empty((1,1,D,D), requires_grad = True).to(device)) 
        self.Wa_sub = nn.Parameter(torch.empty((1,D+F1,D), requires_grad = True).to(device))
        nn.init.normal_(self.Wa_sub), nn.init.normal_(self.Wm1_sub), nn.init.normal_(self.Wm2_sub), nn.init.normal_(self.Wi_sub)
        self.BN1_sub = nn.BatchNorm2d(D).to(device)
        self.BN2_sub = nn.BatchNorm2d(D).to(device)
        
        self.Wi_pro = nn.Parameter(torch.empty((1,1,F,D), requires_grad = True).to(device))
        self.Wm1_pro = nn.Parameter(torch.empty((1,1,D,D), requires_grad = True).to(device)) 
        self.Wm2_pro= nn.Parameter(torch.empty((1,1,D,D), requires_grad = True).to(device)) 
        self.Wa_pro = nn.Parameter(torch.empty((1,D+F1,D), requires_grad = True).to(device))
        nn.init.normal_(self.Wa_pro), nn.init.normal_(self.Wm1_pro), nn.init.normal_(self.Wm2_pro), nn.init.normal_(self.Wi_pro)
        self.BN1_pro = nn.BatchNorm2d(D).to(device)
        self.BN2_pro = nn.BatchNorm2d(D).to(device)

        
        self.OnesN_N = torch.tensor(np.ones((N,N)), dtype = torch.float32, requires_grad = False).to(device)
        self.Ones1_N = torch.tensor(np.ones((1,N)), dtype = torch.float32, requires_grad = False).to(device)

        self.D = D
        #seconda head
        
        self.BN3 = nn.BatchNorm1d(D*2).to(device)
        self.linear1 = nn.Linear(D*2, 32).to(device)
        self.linear2 = nn.Linear(32, 1).to(device)
        
        #dropout_layer
        self.drop_layer = nn.Dropout(p= droprate)

    def forward(self, XE_subs, XE_pros, X_subs, X_pros, A_subs, A_pros, structure_array):
        no_subs, no_pros = int(structure_array[0]), int(structure_array[1])
        
        for i in range(no_subs):
            XE_sub = torch.index_select(XE_subs, 1, torch.tensor([i]).to(device)).view((-1,N,N,F1+F2))
            X_sub = torch.index_select(X_subs, 1, torch.tensor([i]).to(device))
            A_sub = torch.index_select(A_subs, 1, torch.tensor([i]).to(device)).view((-1,N,N,1))
            X_sub = X_sub.view((-1, N, 1, F1))
        
            H0_sub = nn.ReLU()(torch.matmul(XE_sub, self.Wi_sub)) #W*XE
            #only get neighbors in each row: (elementwise multiplication)
            M1_sub = torch.mul(H0_sub, A_sub)
            M1_sub = torch.transpose(M1_sub, dim0 =1, dim1 =2)
            M1_sub = torch.matmul(self.OnesN_N, M1_sub)
            M1_sub = torch.add(M1_sub, -torch.transpose(H0_sub, dim0 =1, dim1 =2) )
            M1_sub = torch.mul(M1_sub, A_sub)
            H1_sub = torch.add(H0_sub, torch.matmul(M1_sub, self.Wm1_sub))
            H1_sub = torch.transpose(H1_sub, dim0 =1, dim1 =3)
            H1_sub = nn.ReLU()(self.BN1_sub(H1_sub))
            H1_sub = torch.transpose(H1_sub, dim0 =1, dim1 =3)


            M2_sub = torch.mul(H1_sub, A_sub)
            M2_sub = torch.transpose(M2_sub, dim0 =1, dim1 =2)
            M2_sub = torch.matmul(self.OnesN_N, M2_sub)
            M2_sub = torch.add(M2_sub, -torch.transpose(H1_sub, dim0 =1, dim1 =2))
            M2_sub = torch.mul(M2_sub, A_sub)
            H2_sub = torch.add(H0_sub, torch.matmul(M2_sub, self.Wm2_sub)) 
            H2_sub = torch.transpose(H2_sub, dim0 =1, dim1 =3)
            H2_sub = nn.ReLU()(self.BN2_sub(H2_sub))
            H2_sub = torch.transpose(H2_sub, dim0 =1, dim1 =3) 

            M_v_sub = torch.mul(H2_sub, A_sub)
            M_v_sub = torch.matmul(self.Ones1_N, M_v_sub)
            XM_sub = torch.cat((X_sub, M_v_sub),3)
            H_sub = nn.ReLU()(torch.matmul(XM_sub, self.Wa_sub))
            h_sub = torch.matmul(self.Ones1_N, torch.transpose(H_sub, dim0 =1, dim1 =2))
            h_sub = self.drop_layer(h_sub.view((-1,self.D)))
            
            if i ==0:
                h_sub_max = h_sub.view(-1,1,self.D)
            else:
                h_sub_max, _= torch.max(torch.cat((h_sub.view(-1,1,self.D), h_sub_max.view(-1,1,self.D)), 1 ), 1, keepdim = True)
                
                
        for i in range(no_pros):
            XE_pro = torch.index_select(XE_pros, 1, torch.tensor([i]).to(device)).view((-1,N,N,F1+F2))
            X_pro = torch.index_select(X_pros, 1, torch.tensor([i]).to(device))
            A_pro = torch.index_select(A_pros, 1, torch.tensor([i]).to(device)).view((-1,N,N,1))
            X_pro = X_pro.view((-1, N, 1, F1))
        
            H0_pro = nn.ReLU()(torch.matmul(XE_pro, self.Wi_pro)) #W*XE
            #only get neighbors in each row: (elementwise multiplication)
            M1_pro = torch.mul(H0_pro, A_pro)
            M1_pro = torch.transpose(M1_pro, dim0 =1, dim1 =2)
            M1_pro = torch.matmul(self.OnesN_N, M1_pro)
            M1_pro = torch.add(M1_pro, -torch.transpose(H0_pro, dim0 =1, dim1 =2) )
            M1_pro = torch.mul(M1_pro, A_pro)
            H1_pro = torch.add(H0_pro, torch.matmul(M1_pro, self.Wm1_pro))
            H1_pro = torch.transpose(H1_pro, dim0 =1, dim1 =3)
            H1_pro = nn.ReLU()(self.BN1_pro(H1_pro))
            H1_pro = torch.transpose(H1_pro, dim0 =1, dim1 =3)


            M2_pro = torch.mul(H1_pro, A_pro)
            M2_pro = torch.transpose(M2_pro, dim0 =1, dim1 =2)
            M2_pro = torch.matmul(self.OnesN_N, M2_pro)
            M2_pro = torch.add(M2_pro, -torch.transpose(H1_pro, dim0 =1, dim1 =2))
            M2_pro = torch.mul(M2_pro, A_pro)
            H2_pro = torch.add(H0_pro, torch.matmul(M2_pro, self.Wm2_pro)) 
            H2_pro = torch.transpose(H2_pro, dim0 =1, dim1 =3)
            H2_pro = nn.ReLU()(self.BN2_pro(H2_pro))
            H2_pro = torch.transpose(H2_pro, dim0 =1, dim1 =3) 

            M_v_pro = torch.mul(H2_pro, A_pro)
            M_v_pro = torch.matmul(self.Ones1_N, M_v_pro)
            XM_pro = torch.cat((X_pro, M_v_pro),3)
            H_pro = nn.ReLU()(torch.matmul(XM_pro, self.Wa_pro))
            h_pro = torch.matmul(self.Ones1_N, torch.transpose(H_pro, dim0 =1, dim1 =2))
            h_pro = self.drop_layer(h_pro.view((-1,self.D)))
            
            if i ==0:
                h_pro_max = h_pro.view(-1,1,self.D)
            else:
                h_pro_max, _= torch.max(torch.cat((h_pro.view(-1,1,self.D), h_pro_max.view(-1,1,self.D)),1 ), 1, keepdim = True)

        
        h = torch.cat((h_sub_max.view(-1,self.D), h_pro_max.view(-1,self.D)),1)
        h =  nn.ReLU()(self.linear1(self.BN3(h)))
        y = self.linear2(h)
        return(y)

    def get_rep(self, XE_subs, XE_pros, X_subs, X_pros, A_subs, A_pros, structure_array):
        no_subs, no_pros = int(structure_array[0]), int(structure_array[1])
        
        for i in range(no_subs):
            XE_sub = torch.index_select(XE_subs, 1, torch.tensor([i]).to(device)).view((-1,N,N,F1+F2))
            X_sub = torch.index_select(X_subs, 1, torch.tensor([i]).to(device))
            A_sub = torch.index_select(A_subs, 1, torch.tensor([i]).to(device)).view((-1,N,N,1))
            X_sub = X_sub.view((-1, N, 1, F1))
        
            H0_sub = nn.ReLU()(torch.matmul(XE_sub, self.Wi_sub)) #W*XE
            #only get neighbors in each row: (elementwise multiplication)
            M1_sub = torch.mul(H0_sub, A_sub)
            M1_sub = torch.transpose(M1_sub, dim0 =1, dim1 =2)
            M1_sub = torch.matmul(self.OnesN_N, M1_sub)
            M1_sub = torch.add(M1_sub, -torch.transpose(H0_sub, dim0 =1, dim1 =2) )
            M1_sub = torch.mul(M1_sub, A_sub)
            H1_sub = torch.add(H0_sub, torch.matmul(M1_sub, self.Wm1_sub))
            H1_sub = torch.transpose(H1_sub, dim0 =1, dim1 =3)
            H1_sub = nn.ReLU()(self.BN1_sub(H1_sub))
            H1_sub = torch.transpose(H1_sub, dim0 =1, dim1 =3)


            M2_sub = torch.mul(H1_sub, A_sub)
            M2_sub = torch.transpose(M2_sub, dim0 =1, dim1 =2)
            M2_sub = torch.matmul(self.OnesN_N, M2_sub)
            M2_sub = torch.add(M2_sub, -torch.transpose(H1_sub, dim0 =1, dim1 =2))
            M2_sub = torch.mul(M2_sub, A_sub)
            H2_sub = torch.add(H0_sub, torch.matmul(M2_sub, self.Wm2_sub)) 
            H2_sub = torch.transpose(H2_sub, dim0 =1, dim1 =3)
            H2_sub = nn.ReLU()(self.BN2_sub(H2_sub))
            H2_sub = torch.transpose(H2_sub, dim0 =1, dim1 =3) 

            M_v_sub = torch.mul(H2_sub, A_sub)
            M_v_sub = torch.matmul(self.Ones1_N, M_v_sub)
            XM_sub = torch.cat((X_sub, M_v_sub),3)
            H_sub = nn.ReLU()(torch.matmul(XM_sub, self.Wa_sub))
            h_sub = torch.matmul(self.Ones1_N, torch.transpose(H_sub, dim0 =1, dim1 =2))
            h_sub = self.drop_layer(h_sub.view((-1,self.D)))
            
            if i ==0:
                h_sub_max = h_sub.view(-1,1,self.D)
            else:
                h_sub_max, _= torch.max(torch.cat((h_sub.view(-1,1,self.D), h_sub_max.view(-1,1,self.D)), 1 ), 1, keepdim = True)
                
                
        for i in range(no_pros):
            XE_pro = torch.index_select(XE_pros, 1, torch.tensor([i]).to(device)).view((-1,N,N,F1+F2))
            X_pro = torch.index_select(X_pros, 1, torch.tensor([i]).to(device))
            A_pro = torch.index_select(A_pros, 1, torch.tensor([i]).to(device)).view((-1,N,N,1))
            X_pro = X_pro.view((-1, N, 1, F1))
        
            H0_pro = nn.ReLU()(torch.matmul(XE_pro, self.Wi_pro)) #W*XE
            #only get neighbors in each row: (elementwise multiplication)
            M1_pro = torch.mul(H0_pro, A_pro)
            M1_pro = torch.transpose(M1_pro, dim0 =1, dim1 =2)
            M1_pro = torch.matmul(self.OnesN_N, M1_pro)
            M1_pro = torch.add(M1_pro, -torch.transpose(H0_pro, dim0 =1, dim1 =2) )
            M1_pro = torch.mul(M1_pro, A_pro)
            H1_pro = torch.add(H0_pro, torch.matmul(M1_pro, self.Wm1_pro))
            H1_pro = torch.transpose(H1_pro, dim0 =1, dim1 =3)
            H1_pro = nn.ReLU()(self.BN1_pro(H1_pro))
            H1_pro = torch.transpose(H1_pro, dim0 =1, dim1 =3)


            M2_pro = torch.mul(H1_pro, A_pro)
            M2_pro = torch.transpose(M2_pro, dim0 =1, dim1 =2)
            M2_pro = torch.matmul(self.OnesN_N, M2_pro)
            M2_pro = torch.add(M2_pro, -torch.transpose(H1_pro, dim0 =1, dim1 =2))
            M2_pro = torch.mul(M2_pro, A_pro)
            H2_pro = torch.add(H0_pro, torch.matmul(M2_pro, self.Wm2_pro)) 
            H2_pro = torch.transpose(H2_pro, dim0 =1, dim1 =3)
            H2_pro = nn.ReLU()(self.BN2_pro(H2_pro))
            H2_pro = torch.transpose(H2_pro, dim0 =1, dim1 =3) 

            M_v_pro = torch.mul(H2_pro, A_pro)
            M_v_pro = torch.matmul(self.Ones1_N, M_v_pro)
            XM_pro = torch.cat((X_pro, M_v_pro),3)
            H_pro = nn.ReLU()(torch.matmul(XM_pro, self.Wa_pro))
            h_pro = torch.matmul(self.Ones1_N, torch.transpose(H_pro, dim0 =1, dim1 =2))
            h_pro = self.drop_layer(h_pro.view((-1,self.D)))
            
            if i ==0:
                h_pro_max = h_pro.view(-1,1,self.D)
            else:
                h_pro_max, _= torch.max(torch.cat((h_pro.view(-1,1,self.D), h_pro_max.view(-1,1,self.D)),1 ), 1, keepdim = True)

        
        h = torch.cat((h_sub_max.view(-1,self.D), h_pro_max.view(-1,self.D)),1)
        return(h)



class GNN_mean(nn.Module):
    def __init__(self, D= 30, N = 70, F1 = 32 , F2 = 10, F = F1+F2, droprate = 0.2):
        super(GNN, self).__init__()
        #first head
        self.Wi_sub = nn.Parameter(torch.empty((1,1,F,D), requires_grad = True).to(device))
        self.Wm1_sub = nn.Parameter(torch.empty((1,1,D,D), requires_grad = True).to(device)) 
        self.Wm2_sub= nn.Parameter(torch.empty((1,1,D,D), requires_grad = True).to(device)) 
        self.Wa_sub = nn.Parameter(torch.empty((1,D+F1,D), requires_grad = True).to(device))
        nn.init.normal_(self.Wa_sub), nn.init.normal_(self.Wm1_sub), nn.init.normal_(self.Wm2_sub), nn.init.normal_(self.Wi_sub)
        self.BN1_sub = nn.BatchNorm2d(D).to(device)
        self.BN2_sub = nn.BatchNorm2d(D).to(device)
        
        self.Wi_pro = nn.Parameter(torch.empty((1,1,F,D), requires_grad = True).to(device))
        self.Wm1_pro = nn.Parameter(torch.empty((1,1,D,D), requires_grad = True).to(device)) 
        self.Wm2_pro= nn.Parameter(torch.empty((1,1,D,D), requires_grad = True).to(device)) 
        self.Wa_pro = nn.Parameter(torch.empty((1,D+F1,D), requires_grad = True).to(device))
        nn.init.normal_(self.Wa_pro), nn.init.normal_(self.Wm1_pro), nn.init.normal_(self.Wm2_pro), nn.init.normal_(self.Wi_pro)
        self.BN1_pro = nn.BatchNorm2d(D).to(device)
        self.BN2_pro = nn.BatchNorm2d(D).to(device)

        
        self.OnesN_N = torch.tensor(np.ones((N,N)), dtype = torch.float32, requires_grad = False).to(device)
        self.Ones1_N = torch.tensor(np.ones((1,N)), dtype = torch.float32, requires_grad = False).to(device)

        self.D = D
        #seconda head
        
        self.BN3 = nn.BatchNorm1d(D*2).to(device)
        self.linear1 = nn.Linear(D*2, 32).to(device)
        self.linear2 = nn.Linear(32, 1).to(device)
        
        #dropout_layer
        self.drop_layer = nn.Dropout(p= droprate)

    def forward(self, XE_subs, XE_pros, X_subs, X_pros, A_subs, A_pros, structure_array):
        no_subs, no_pros = int(structure_array[0]), int(structure_array[1])
        
        for i in range(no_subs):
            XE_sub = torch.index_select(XE_subs, 1, torch.tensor([i]).to(device)).view((-1,N,N,F1+F2))
            X_sub = torch.index_select(X_subs, 1, torch.tensor([i]).to(device))
            A_sub = torch.index_select(A_subs, 1, torch.tensor([i]).to(device)).view((-1,N,N,1))
            X_sub = X_sub.view((-1, N, 1, F1))
        
            H0_sub = nn.ReLU()(torch.matmul(XE_sub, self.Wi_sub)) #W*XE
            #only get neighbors in each row: (elementwise multiplication)
            M1_sub = torch.mul(H0_sub, A_sub)
            M1_sub = torch.transpose(M1_sub, dim0 =1, dim1 =2)
            M1_sub = torch.matmul(self.OnesN_N, M1_sub)
            M1_sub = torch.add(M1_sub, -torch.transpose(H0_sub, dim0 =1, dim1 =2) )
            M1_sub = torch.mul(M1_sub, A_sub)
            H1_sub = torch.add(H0_sub, torch.matmul(M1_sub, self.Wm1_sub))
            H1_sub = torch.transpose(H1_sub, dim0 =1, dim1 =3)
            H1_sub = nn.ReLU()(self.BN1_sub(H1_sub))
            H1_sub = torch.transpose(H1_sub, dim0 =1, dim1 =3)


            M2_sub = torch.mul(H1_sub, A_sub)
            M2_sub = torch.transpose(M2_sub, dim0 =1, dim1 =2)
            M2_sub = torch.matmul(self.OnesN_N, M2_sub)
            M2_sub = torch.add(M2_sub, -torch.transpose(H1_sub, dim0 =1, dim1 =2))
            M2_sub = torch.mul(M2_sub, A_sub)
            H2_sub = torch.add(H0_sub, torch.matmul(M2_sub, self.Wm2_sub)) 
            H2_sub = torch.transpose(H2_sub, dim0 =1, dim1 =3)
            H2_sub = nn.ReLU()(self.BN2_sub(H2_sub))
            H2_sub = torch.transpose(H2_sub, dim0 =1, dim1 =3) 

            M_v_sub = torch.mul(H2_sub, A_sub)
            M_v_sub = torch.matmul(self.Ones1_N, M_v_sub)
            XM_sub = torch.cat((X_sub, M_v_sub),3)
            H_sub = nn.ReLU()(torch.matmul(XM_sub, self.Wa_sub))
            h_sub = torch.matmul(self.Ones1_N, torch.transpose(H_sub, dim0 =1, dim1 =2))
            h_sub = self.drop_layer(h_sub.view((-1,self.D)))
            if i ==0:
                h_sub_max = h_sub.view(-1,self.D)
            else:
                h_sub_max = h_sub_max.add(h_sub)
        h_sub_max = torch.div(h_sub_max, no_subs)

        for i in range(no_pros):
            XE_pro = torch.index_select(XE_pros, 1, torch.tensor([i]).to(device)).view((-1,N,N,F1+F2))
            X_pro = torch.index_select(X_pros, 1, torch.tensor([i]).to(device))
            A_pro = torch.index_select(A_pros, 1, torch.tensor([i]).to(device)).view((-1,N,N,1))
            X_pro = X_pro.view((-1, N, 1, F1))
        
            H0_pro = nn.ReLU()(torch.matmul(XE_pro, self.Wi_pro)) #W*XE
            #only get neighbors in each row: (elementwise multiplication)
            M1_pro = torch.mul(H0_pro, A_pro)
            M1_pro = torch.transpose(M1_pro, dim0 =1, dim1 =2)
            M1_pro = torch.matmul(self.OnesN_N, M1_pro)
            M1_pro = torch.add(M1_pro, -torch.transpose(H0_pro, dim0 =1, dim1 =2) )
            M1_pro = torch.mul(M1_pro, A_pro)
            H1_pro = torch.add(H0_pro, torch.matmul(M1_pro, self.Wm1_pro))
            H1_pro = torch.transpose(H1_pro, dim0 =1, dim1 =3)
            H1_pro = nn.ReLU()(self.BN1_pro(H1_pro))
            H1_pro = torch.transpose(H1_pro, dim0 =1, dim1 =3)


            M2_pro = torch.mul(H1_pro, A_pro)
            M2_pro = torch.transpose(M2_pro, dim0 =1, dim1 =2)
            M2_pro = torch.matmul(self.OnesN_N, M2_pro)
            M2_pro = torch.add(M2_pro, -torch.transpose(H1_pro, dim0 =1, dim1 =2))
            M2_pro = torch.mul(M2_pro, A_pro)
            H2_pro = torch.add(H0_pro, torch.matmul(M2_pro, self.Wm2_pro)) 
            H2_pro = torch.transpose(H2_pro, dim0 =1, dim1 =3)
            H2_pro = nn.ReLU()(self.BN2_pro(H2_pro))
            H2_pro = torch.transpose(H2_pro, dim0 =1, dim1 =3) 

            M_v_pro = torch.mul(H2_pro, A_pro)
            M_v_pro = torch.matmul(self.Ones1_N, M_v_pro)
            XM_pro = torch.cat((X_pro, M_v_pro),3)
            H_pro = nn.ReLU()(torch.matmul(XM_pro, self.Wa_pro))
            h_pro = torch.matmul(self.Ones1_N, torch.transpose(H_pro, dim0 =1, dim1 =2))
            h_pro = self.drop_layer(h_pro.view((-1,self.D)))
            
            if i ==0:
                h_pro_max = h_pro.view(-1,self.D)
            else:
                h_pro_max = h_pro_max.add(h_pro)
        h_pro_max = torch.div(h_pro_max, no_pros)
        
        print(h_pro_max.shape, h_sub_max.shape)
        
        h = torch.cat((h_sub_max.view(-1,self.D), h_pro_max.view(-1,self.D)),1)
        h =  nn.ReLU()(self.linear1(self.BN3(h)))
        y = self.linear2(h)
        return(y)