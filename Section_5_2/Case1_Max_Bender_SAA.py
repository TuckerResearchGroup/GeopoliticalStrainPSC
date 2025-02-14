# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:15:39 2024

@author:  Martha Sabogal
"""

import multiprocessing
from multiprocessing import Pool
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import scipy.stats as st 
import math
import time
import random

#%% Sets&Data   

ini_time = time.time()

sets_df = pd.read_excel('Data.xlsx', sheet_name='sets')

I = sets_df['I'].dropna().tolist()      # supplier countries
J = sets_df['J'].dropna().tolist()      # potential partner countries
K = sets_df['K'].dropna().tolist()      # demand points

F_df = pd.read_excel('Data.xlsx', sheet_name='Fk') 
    
F, Fprime = {}, {}
for k in K:
    F[k] = F_df[k].dropna().tolist()                    # Set of allies of every country without k
    Fprime[k] = list(set(K)-set(F[k]))
for k in K:
    Fprime[k].remove(k)                                 # Set of non-allies of every country without k

JnF, JnFprime = {},{}                                   
for i in I:
    JnF[i] = list(set(F[i]).intersection(J))                # Allied plants of supplier i
    JnFprime[i] = list(set(Fprime[i]).intersection(J))      # Non-allied plants of supplier i

numM = 30       # number of replications
W = 300         # number of scenarios
numE = 3000     # number of scenarios for evaluation

slope = 0.00003     # slope of opportunity cost

# suppliers parameters
supplier_df = pd.read_excel('Data.xlsx', sheet_name='I')
ISOS, c_rm, q_sc = gp.multidict(supplier_df[['ISO','c_rm','q_sc']].set_index('ISO').T.to_dict('list'))   # create dictionario for raw material cost and capacity supplier

total_sc = sum(q_sc[i] for i in I)

# plants parameters
plant_df = pd.read_excel('Data.xlsx', sheet_name='J')
ISOP, c_pr, c_fi, q_pc = gp.multidict(plant_df[['ISO','c_pr','c_fi','q_pc']].set_index('ISO').T.to_dict('list'))  

# demand points parameters
client_df = pd.read_excel('Data.xlsx', sheet_name='K')
ISOD, c_s, e_F, e_Fprime = gp.multidict(client_df[['ISO','c_s', 'e_F', 'e_Fprime']].set_index('ISO').T.to_dict('list'))

p_prime = client_df[['ISO','xi_Fprime']].set_index('ISO').T.to_dict('list')                # p value of Ber Dist for non-allies
p = client_df[['ISO','xi_F']].set_index('ISO').T.to_dict('list')                           # p value of Ber Dist for allies

d = client_df[['ISO','xi_d m','xi_d stv']].set_index('ISO').T.to_dict('list')              # mean and stv of Normal Dist

# transportation parameters
c_t1_df = pd.read_excel('Data.xlsx', sheet_name='c_t1')
c_t1_df.rename(columns = {'Unnamed: 0':'ISO'}, inplace=True)    
dft1 = c_t1_df.melt(id_vars=["ISO"], var_name="To", value_name="c_t1")   
dft1 = dft1.set_index(['ISO','To'])                                      
c_t1 = dict(zip(dft1.index, dft1.c_t1))                                  

c_t2_df = pd.read_excel('Data.xlsx', sheet_name='c_t2')
c_t2_df.rename(columns = {'Unnamed: 0':'ISO'}, inplace=True)
dft2 = c_t2_df.melt(id_vars=["ISO"], var_name="To", value_name="c_t2")
dft2 = dft2.set_index(['ISO','To'])
c_t2 = dict(zip(dft2.index, dft2.c_t2))

# probability mass function for strain supplier capacities
cap_ava = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
supplier_df2 = supplier_df[['ISO']+cap_ava].set_index('ISO')          
p_sd = supplier_df[['ISO','psd']].set_index('ISO').T.to_dict('list')  # p value of Ber Dist of supplier disruption

# probability mass function for production capacities
plant_df2 = plant_df[['ISO']+cap_ava].set_index('ISO')                
p_pd = plant_df[['ISO','ppd']].set_index('ISO').T.to_dict('list')     # p value of Ber Dist of production disruption

#%% Stochastic Data 

def gen_scenarios(d, total_sc, supplier_df2, cap_ava, plant_df2, p, p_prime, p_sd, p_pd, W):              # Sampling, Random parameters
    xi_sc, xi_pc, xi_d, xi_Fprime, xi_F, xi_ss, xi_sd, xi_ps, xi_pd = {}, {}, {}, {}, {}, {}, {}, {}, {}
    xi_dfinal = {}
    
    for w in range(W):        
        xi_tsc = 0
        for i in I:
            sprobabilities = [supplier_df2.loc[i, c] for c in cap_ava]       
            xi_ss[i,w] = np.random.choice(cap_ava, 1, p=sprobabilities)[0]   # strains supplier capacity
            xi_sd[i,w] = np.random.binomial(1, p_sd[i][0])                   # disruption of supplier capacity (1 operating, 0 o/w)
            xi_sc[i,w] = xi_ss[i,w]*xi_sd[i,w]                               # supplier capacity available
            xi_tsc += q_sc[i]*xi_sc[i,w]                                     # total supplier capacity available in scenario 
        
        for j in J:
            pprobabilities = [plant_df2.loc[j, c] for c in cap_ava]         
            xi_ps[j,w] = np.random.choice(cap_ava, 1, p=pprobabilities)[0]  # strains in production capacity
            xi_pd[j,w] = np.random.binomial(1, p_pd[j][0])                  # disruption of production capacity (1 operating, 0 o/w)
            xi_pc[j,w] = xi_ps[j,w]*xi_pd[j,w]                              # production capacity available
         
        if (xi_tsc/total_sc) < 0.8:                                         # if sc is less than 80%, then there is export risk, conditional part
            for k in K:            
                xi_Fprime[k,w] = np.random.binomial(1, p_prime[k][0])       # export risk for non-allies
                if xi_Fprime[k,w] == 0:                                     # if the country is open, it is open to allies, if it is closed, it can be open or closed to allies
                    xi_F[k,w] = np.random.binomial(1, p[k][0]) 
                else:
                    xi_F[k,w] = 1 
                    
        else:                                                               # if sc is >= 80% then there is no risk of export ban
            for k in K: 
                xi_Fprime[k,w] = 1                                          
                xi_F[k,w] = 1                                              
                
        for k in K:
            xi_d[k,w] = np.random.normal(d[k][0], d[k][1])                  # demand
            if xi_d[k,w] < 0:
                xi_d[k,w] = 0
            
            xi_dfinal[k,w] = max(xi_d[k,w]-(1-xi_Fprime[k,w])*e_Fprime[k]-(1-xi_F[k,w])*e_F[k], 0)
                  
    return xi_sc, xi_pc, xi_dfinal, xi_F, xi_Fprime, xi_d

def get_co (slope, e_F, e_Fprime, xi_F, xi_Fprime, w):
    G = sum(e_Fprime[k]*(1-xi_Fprime[k,w]) + e_F[k]*(1-xi_F[k,w]) for k in K)   # global shortage because of export bans
    c_o = slope*G                                                               # total opportunity cost because of ban export
    
    return c_o

xi_sc, xi_pc, xi_dfinal, xi_F, xi_Fprime, xi_d = gen_scenarios(d, total_sc, supplier_df2, cap_ava, plant_df2, p, p_prime, p_sd, p_pd, W)     # Random Variables
c_o = get_co(slope, e_F, e_Fprime, xi_F, xi_Fprime, 0)                          # Opportunity Cost


#%% To store results

LBSAA = -1e20                                               # Initialization Lower bound of SAA
UBSAA = 1e20                                                # Initialization Upper bound of SAA
meanS, meanPS, meanV, meanU, meanPSR = {}, {}, {}, {}, {}   # statistics 
SE, PSE, PSRE = {}, {}, {}                                  # to capture shortage, and percentage of shortage in evaluation sample
GS = np.zeros(numE)                                         # to capture global shortage in scenario w
GD = np.zeros(numE)                                         # global demand in scenario w (consider lost demand)
GRD = np.zeros(numE)                                        # global actual demand in scenario w 
co = np.zeros(numE)                                         # capture opportunity cost
drug_cost = np.zeros(numE)                                  # pn + tr2
rm_cost = np.zeros(numE)                                    # rm + tr1
ssof_hat = np.zeros(numE)                                   # second stage OF    
co_income = np.zeros(numE)                                  # income due to co
cs_income = np.zeros(numE)                                  # income due to cs
sales_cost = np.zeros(numE)                                 # rm costs + drug costs
sales_vol = np.zeros(numE)                                  # total V per scenario w

# Initialize statistics    
for w in range(numE):
    for k in K:   
        SE[k,w] = 0
        PSE[k,w] = 0
for j in J:
    for k in K:
        meanV[j,k] = 0
    for i in I:
        meanU[i,j] = 0

#%% Master
ini_time_models = time.time()

m_mp = gp.Model('Two-Stage-MP')

# Decision Variables
Y = {}      
for j in J:
    Y[j] = m_mp.addVar(vtype=GRB.BINARY, obj=-c_fi[j]) 

theta = m_mp.addVar(vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = 1e20, obj=1)

# Objective Function 
m_mp.ModelSense = GRB.MAXIMIZE

# make at least one alliance
c1_1 = m_mp.addConstr(gp.quicksum(Y[j] for j in J) >=1, "c1_1")

m_mp.update()

cut = {}

#%% Subproblem
m_subp = gp.Model('Two-Stage-SubP')

# Decision Variables
U, V, S = {}, {}, {}

for j in J:
    for i in I:
        U[i,j] = m_subp.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
    for k in K:
        V[j,k] = m_subp.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
for k in K:
    S[k] = m_subp.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)

# Objective function
m_subp.setObjective(gp.quicksum(gp.quicksum((c_s[k]-c_pr[j]-c_t2[j,k])*V[j,k] for k in K) for j in J)+ 
                    gp.quicksum(c_o*xi_Fprime[j,1]*V[j,j] + gp.quicksum(c_o*V[j,k] for k in K if k!=j) for j in J)-
                    gp.quicksum(gp.quicksum((c_rm[i]+c_t1[i,j])*U[i,j] for j in J) for i in I) , gp.GRB.MAXIMIZE)

# Constraints
c2_1, c2_2, c2_3, c2_4, c2_5, c2_6, c2_7, c2_8 = {}, {}, {}, {}, {}, {}, {}, {}
        
# raw material purchases - supplier capacity      
for i in I:
    c2_1[i] = m_subp.addConstr(gp.quicksum(U[i,j] for j in J) <= q_sc[i]*xi_sc[i,1])
    
# raw material purchases - supplier capacity and export ban for non-allies      
for i in I:
    for j in JnFprime[i]:
        c2_2[i,j] = m_subp.addConstr(U[i,j] <= q_sc[i]*xi_sc[i,1]*xi_Fprime[i,1])    

# raw material purchases - supplier capacity and export ban for allies     
for i in I:
    for j in JnF[i]:
        c2_3[i,j] = m_subp.addConstr(U[i,j] <= q_sc[i]*xi_sc[i,1]*xi_F[i,1]) 

# production - manufacturing capacity
for j in J:
    c2_4[j] = m_subp.addConstr(gp.quicksum(V[j,k] for k in K) <= q_pc[j]*xi_pc[j,1])
    
# production - manufacturing capacity and export ban for non-allies      
for j in J:
    for k in Fprime[j]:
        c2_5[j,k] = m_subp.addConstr(V[j,k] <= q_pc[j]*xi_pc[j,1]*xi_Fprime[j,1])    
    
# production - manufacturing capacity and export ban for allies     
for j in J:
    for k in F[j]:
        c2_6[j,k] = m_subp.addConstr(V[j,k] <= q_pc[j]*xi_pc[j,1]*xi_F[j,1])  
           
# demand constraint
for k in K:
    c2_7[k] = m_subp.addConstr(gp.quicksum(V[j,k]for j in J)+S[k] == xi_dfinal[k,1])
         
# flow balance
for j in J:
    c2_8[j] = m_subp.addConstr(gp.quicksum(U[i,j] for i in I) - gp.quicksum(V[j,k] for k in K) == 0)
    
m_subp.update()

m_mp.setParam("OutputFlag", 0)      
m_subp.setParam("OutputFlag", 0)    

#%% Functions

# Create dual variables
dualc2_1, dualc2_2, dualc2_3, dualc2_4, dualc2_5, dualc2_6, dualc2_7, dualc2_8 = {}, {}, {}, {}, {}, {}, {}, {}

for w in range(W):   
    for i in I:
        dualc2_1[i,w] = 0
        for j in JnFprime[i]:
            dualc2_2[i,j,w] = 0  
        for j in JnF[i]: 
            dualc2_3[i,j,w] = 0 
    for j in J:
        dualc2_4[j,w] = 0
        for k in Fprime[j]:
            dualc2_5[j,k,w] = 0  
        for k in F[j]:
            dualc2_6[j,k,w] = 0  
    for k in K:
        dualc2_7[k,w] = 0  
    for j in J:
        dualc2_8[j,w] = 0      

# Capture dual variables
def get_duals (w):
    for i in I:
        dualc2_1[i,w] = c2_1[i].pi
        for j in JnFprime[i]:
            dualc2_2[i,j,w] = c2_2[i,j].pi   
        for j in JnF[i]: 
            dualc2_3[i,j,w] = c2_3[i,j].pi  
    for j in J:
        dualc2_4[j,w] = c2_4[j].pi
        for k in Fprime[j]:
            dualc2_5[j,k,w] = c2_5[j,k].pi   
        for k in F[j]:
            dualc2_6[j,k,w] = c2_6[j,k].pi
    for k in K:
        dualc2_7[k,w] = c2_7[k].pi
    for j in J:
        dualc2_8[j,w] = c2_8[j].pi         
    
    return dualc2_1, dualc2_2, dualc2_3, dualc2_4, dualc2_5, dualc2_6, dualc2_7, dualc2_8

# Update the RHS
def update_rhs (q_sc, q_pc, xi_sc, xi_pc, xi_dfinal, xi_Fprime, xi_F, Yval, w):
    for i in I:
        c2_1[i].setAttr(GRB.Attr.RHS, q_sc[i]*xi_sc[i,w])
        for j in JnFprime[i]:
            c2_2[i,j].setAttr(GRB.Attr.RHS, q_sc[i]*xi_sc[i,w]*xi_Fprime[i,w]*Yval[j])    
        for j in JnF[i]:
            c2_3[i,j].setAttr(GRB.Attr.RHS, q_sc[i]*xi_sc[i,w]*xi_F[i,w]*Yval[j]) 
    for j in J:
        c2_4[j].setAttr(GRB.Attr.RHS, q_pc[j]*xi_pc[j,w]*Yval[j])
        for k in Fprime[j]:
            c2_5[j,k].setAttr(GRB.Attr.RHS, q_pc[j]*xi_pc[j,w]*xi_Fprime[j,w]*Yval[j]) 
        for k in F[j]:
            c2_6[j,k].setAttr(GRB.Attr.RHS, q_pc[j]*xi_pc[j,w]*xi_F[j,w]*Yval[j])
    for k in K:
        c2_7[k].setAttr(GRB.Attr.RHS, xi_dfinal[k,w])
    for j in J:
        c2_8[j].setAttr(GRB.Attr.RHS, 0)
        
    m_subp.update()

#%% SAA

def f_rep(m):
    ini_timeR = time.time()                                 # start time of the replication
    f = open("Console.txt","a")                             # create txt file to store replication results
    v = random.randint(1,100000000)                         # to have different seed every time a run the code
    np.random.seed(m*v)                                     # to have different seed in every replication
    xi_sc, xi_pc, xi_dfinal, xi_F, xi_Fprime, xi_d = gen_scenarios(d, total_sc, supplier_df2, cap_ava, plant_df2, p, p_prime, p_sd, p_pd, W)   # create sample
    
    m_mp.update()
    
    ## L-SHAPED METHOD ----------------------------------------------------------------------------------------------------------------------
    print ('************Benders - L Shaped method - Single Cut***************** \n')
    print('Replication', m)
    
    LowB = -1e20
    UpperB = 1e20
    iter = 0
           
    while (UpperB - LowB)/UpperB > 1e-5:                    # optimality criteria
        iter += 1
        m_mp.update()
        m_mp.optimize()     # solve MP
       
        Yval={}             # store solution of MP
        for j in J:
            Yval[j] = Y[j].x
        thetaval = theta.x
        
        print("\n""Iteration: ", iter, "Objective MP (UB): ", m_mp.objVal)
        
        UpperB = m_mp.objVal
        CX = thetaval - UpperB
        tempLB = 0        
                        
        for w in range(W): 
            c_o = get_co(slope, e_F, e_Fprime, xi_F, xi_Fprime, w)                                      # calculate opportunity cost
            update_rhs(q_sc, q_pc, xi_sc, xi_pc, xi_dfinal, xi_Fprime, xi_F, Yval, w)                   # update parameters in RHS            
            
            m_subp.setObjective(gp.quicksum(gp.quicksum((c_s[k]-c_pr[j]-c_t2[j,k])*V[j,k] for k in K) for j in J)+ 
                    gp.quicksum(c_o*xi_Fprime[j,w]*V[j,j] + gp.quicksum(c_o*V[j,k] for k in K if k!=j) for j in J)-
                    gp.quicksum(gp.quicksum((c_rm[i]+c_t1[i,j])*U[i,j] for j in J) for i in I), gp.GRB.MAXIMIZE)
                       
            # since we have complete recourse, there is no feasibility cuts
            m_subp.update()        
            m_subp.optimize()                                                                           # solve subproblem
    
            tempLB += m_subp.objVal*(1/W)
            get_duals(w)                                                                                # capture dual variables
            
        tempLB = tempLB - CX
        if tempLB > LowB:
            LowB = tempLB
        
        # comparison for convergence
        if (UpperB - LowB)/UpperB <= 1e-5:
            print("\n""**Optimal solution is reached at iteration ",iter)
            break
        
        # if it does not converge, generate a single cut
        else:
            cut[iter] = m_mp.addConstr( theta <= (1/W)*gp.quicksum(gp.quicksum(dualc2_1[i,w]*(q_sc[i]*xi_sc[i,w]) for i in I)+                                           
            gp.quicksum(gp.quicksum(dualc2_2[i,j,w]*(q_sc[i]*xi_sc[i,w]*xi_Fprime[i,w]*Y[j]) for j in JnFprime[i]) for i in I)+
            gp.quicksum(gp.quicksum(dualc2_3[i,j,w]*(q_sc[i]*xi_sc[i,w]*xi_F[i,w]*Y[j]) for j in JnF[i]) for i in I)+
            gp.quicksum(dualc2_4[j,w]*(q_pc[j]*xi_pc[j,w]*Y[j]) for j in J)+
            gp.quicksum(gp.quicksum(dualc2_5[j,k,w]*(q_pc[j]*xi_pc[j,w]*xi_Fprime[j,w]*Y[j]) for k in Fprime[j]) for j in J)+
            gp.quicksum(gp.quicksum(dualc2_6[j,k,w]*(q_pc[j]*xi_pc[j,w]*xi_F[j,w]*Y[j]) for k in F[j]) for j in J)+
            gp.quicksum(dualc2_7[k,w]*(xi_dfinal[k,w]) for k in K)+
            gp.quicksum(dualc2_8[j,w]*0 for j in J) for w in range(W)))
            m_mp.update()
    
    end_timeR = time.time()-ini_timeR                           # execution time of replication 
    
    m_mp.remove(cut)                                               
        
    print("LB: ", LowB)                                         # OF of SUBP adding CX
    print("UB: ", UpperB)                                       # OF of MP
    print('Selected Plants')
    for j in J:
        if Y[j].x > 0:
            print("{}:".format(j),Y[j].x)  
           
    ## Evaluate first-stage decisions in a bigger number of scenarios and select the one that produce the largest objective function --------- 
        
    # Create another sample with >> scenarios
    vv = random.randint(1,100000000)
    np.random.seed(m*vv)
    xi_sc, xi_pc, xi_dfinal, xi_F, xi_Fprime, xi_d = gen_scenarios(d, total_sc, supplier_df2, cap_ava, plant_df2, p, p_prime, p_sd, p_pd, numE) 
                  
    ## Subproblem
    ssof_numE = 0
    
    for w in range(numE):          
        c_o = get_co(slope, e_F, e_Fprime, xi_F, xi_Fprime, w)
        update_rhs(q_sc, q_pc, xi_sc, xi_pc, xi_dfinal, xi_Fprime, xi_F, Yval, w)
        
        m_subp.setObjective(gp.quicksum(gp.quicksum((c_s[k]-c_pr[j]-c_t2[j,k])*V[j,k] for k in K) for j in J)+ 
                    gp.quicksum(c_o*xi_Fprime[j,w]*V[j,j] + gp.quicksum(c_o*V[j,k] for k in K if k!=j) for j in J)-
                    gp.quicksum(gp.quicksum((c_rm[i]+c_t1[i,j])*U[i,j] for j in J) for i in I) , gp.GRB.MAXIMIZE)
                
        m_subp.update()
        m_subp.optimize()
        ssof_numE += m_subp.objVal*(1/numE)
    
    # Evaluating candidate solution
    Z_numE = -CX + ssof_numE
    f.write("\n Replication: {} Optimal Iteration: {} RET: {} Obj Replication Z_N^m: {} Z_N'^m: {}".format(m, iter, end_timeR, UpperB, Z_numE))
    f.close()
    
    return(UpperB,Z_numE, Yval)

#%%MULTIPROCESSING  
if __name__=='__main__':
    objvals = []                                             
    with multiprocessing.Pool(30) as pool:                   # use multiprocessing (Replace 30 with the number of cores available)
        for result in pool.map(f_rep, range(1,numM+1)):      # for every replication will perform the function and store results
            objvals.append(result)
        pool.close()
        pool.join()                                          # cannot start another process without finishing this
    
    #%%SELECT SOLUTION   
    objvals.sort(key=lambda x: x[1], reverse=True)           # sort from largest to smallest column Z_numE                  
    Y_hat = objvals[0][2]                                    # store solution with largest Z_numE value
    UBSAA = np.mean([x[0] for x in objvals])                 # statistical upper bound of optimal OF value of original problem 
    print("\n""mean SAA objval U_NM = ", UBSAA)
    f = open("Aggregate.txt","a")
    
    #%% EVALUATION SAA
    ini_timeE = time.time()  
    print("\n""EVALUATION SAA ")
        
    # Taking the best 1st-stage decision
    CX_hat = 0
    for j in J:    
        CX_hat += c_fi[j]*Y_hat[j]                            
    
    # Create another sample with >> scenarios 
    xi_sc, xi_pc, xi_dfinal, xi_F, xi_Fprime, xi_d = gen_scenarios(d, total_sc, supplier_df2, cap_ava, plant_df2, p, p_prime, p_sd, p_pd, numE)
    
    ## Subproblem
    for w in range(numE):          
        c_o = get_co(slope, e_F, e_Fprime, xi_F, xi_Fprime, w)
        update_rhs(q_sc, q_pc, xi_sc, xi_pc, xi_dfinal, xi_Fprime, xi_F, Y_hat, w)
        
        m_subp.setObjective(gp.quicksum(gp.quicksum((c_s[k]-c_pr[j]-c_t2[j,k])*V[j,k] for k in K) for j in J)+ 
                    gp.quicksum(c_o*xi_Fprime[j,w]*V[j,j] + gp.quicksum(c_o*V[j,k] for k in K if k!=j) for j in J)-
                    gp.quicksum(gp.quicksum((c_rm[i]+c_t1[i,j])*U[i,j] for j in J) for i in I), gp.GRB.MAXIMIZE)
                
        m_subp.update()
        m_subp.optimize()
                
        ssof_hat[w] = m_subp.objVal                             # storing the 2-stage OF
        co[w] = c_o                                             # saving opportunity cost
        for k in K:
            GS[w] += S[k].x                                     # global shortage per scenario
            GD[w] += xi_d[k,w]                                  # global demand per scenario considering lost demand
            GRD[w] += xi_dfinal[k,w]                            # global actual demand per scenario    
            
            SE[k,w] = S[k].x                                    # shortage
            PSE[k,w] = S[k].x/(xi_d[k,w]+0.0000001)             # percentage of shortage
            PSRE[k,w] = S[k].x/(xi_dfinal[k,w]+0.0000001)       # percentage of shortage
           
            for j in J:
                meanV[j,k] += (1/numE)*V[j,k].x   
                cs_income[w] += c_s[k]*V[j,k].x 
                drug_cost[w] += (c_pr[j]+c_t2[j,k])*V[j,k].x 
                sales_vol[w] += V[j,k].x 
                   
        for j in J:
            for i in I:
                meanU[i,j] += (1/numE)*U[i,j].x
                rm_cost[w] += (c_rm[i]+c_t1[i,j])*U[i,j].x
                        
    LBSAA = np.mean(ssof_hat) - CX_hat
    Gap = (UBSAA - LBSAA)/UBSAA            
    end_timeE = (time.time()-ini_timeE)     # evaluation execution time
    
    #%% Metrics and results
    ini_timeD = time.time()
    print("LBSAA Z_N`: ",LBSAA)
    f.write ("\n LBSAA Z_N`:{}".format(LBSAA))
    f.write ("\n UBSAA Z_NM:{}".format(UBSAA))
    f.write ("\n Evaluation Time:{}".format(end_timeE))
    f.write ("\n Fixed cost:{}".format(CX_hat))
    
    if (UBSAA - LBSAA)/UBSAA <= 0.01:
        if (UBSAA - LBSAA)/UBSAA >= 0:
            print("Optimality Gap Goal reached")
            f.write ("\n Optimality Gap Goal reached")
        else:
            print("\n LBSAA > UBSAA")
            f.write ("\n LBSAA > UBSAA")
    else: 
        print("Optimality Gap greater than 1%: ", Gap*100)
        f.write ("\n Optimality Gap greater than 1%")
    
    # Variance of the UBSAA U_NM estimator
    sigma2 = 0
    for m in range(numM):
        sigma2 += (objvals[m][0]-UBSAA)**2    
    sigma2 = sigma2/((numM-1)*numM)
    sigma = math.sqrt(sigma2)
    print("Stdev UBSAA Z_NM estimator = ", sigma) 
    
    # Variance of LBSAA L_N` estimator
    sigma2_hat = 0
    for w in range(numE):
        sigma2_hat += (-CX_hat + ssof_hat[w] - LBSAA)**2
    sigma2_hat = sigma2_hat/((numE-1)*numE)
    sigma_hat = math.sqrt(sigma2_hat)
    print("Stdev LBSAA Z_N` estimator = ", sigma_hat)
    
    # Optimality Gap Estimator
    print("\n""Optimality Gap Z_NM - Z_N`: ", 100*Gap, "%")
    f.write ("\n Optimality Gap% Z_NM - Z_N`:{}".format(100*Gap))
    
    # Variance of Optimality Gap Estimator
    sigma2_gap = sigma2 + sigma2_hat
    sigma_gap = math.sqrt(sigma2_gap)
    print("Stdev Opt. Gap (Z_N`-Z_NM): ", sigma_gap)
    f.write ("\n Stdev of estimator Z_NM: {} Z_N`: {} Opt.Gap(Z_N`-Z_NM): {}".format(sigma, sigma_hat, sigma_gap))
    
    # New bounds with confidence interval
    t = st.t.ppf(q=.99,df=29)
    z = st.norm.ppf(.99)
    LBSAA2 = LBSAA - z*sigma_hat 
    UBSAA2 = UBSAA + t*sigma   
    
    Gap2 = ((UBSAA2 - LBSAA2)/UBSAA2)*100
    f.write ("\n UBSAA2 Z_NM 99%: {} LBSAA2 Z_N`99%: {} Opt.Gap 98%: {}".format(UBSAA2, LBSAA2, Gap2))
    
    # To export data to excel
    meanGS = np.mean(GS)                                        # mean global shortage
    semGS = st.sem(GS)                                          # standard error of the mean global shortage
    print("Mean Global Shortage", meanGS)
    f.write ("\n\n Mean Global Shortage: {} semGS: {} CIGS: {}".format(meanGS, semGS, st.norm.interval(alpha=0.95, loc=meanGS, scale=semGS)))
    
    PGS = np.zeros(numE)
    PGSR = np.zeros(numE)
    for w in range(numE):
        PGS[w] = GS[w]/GD[w]
        PGSR[w] = GS[w]/GRD[w]                                                # percentage of global shortage in every scenario   
        co_income[w] = ssof_hat[w]-cs_income[w]+drug_cost[w]+rm_cost[w]
        sales_cost[w] = drug_cost[w]+rm_cost[w]
    
    meanPGS = np.mean(PGS)                                        # mean global percentage of shortage
    semPGS = st.sem(PGS)                                          # standard error of the mean percentage of global shortage
    print("Mean Percentage of Global Shortage", meanPGS)
    f.write ("\n Mean Percentage of Global Shortage: {} semPGS: {} CIPGS: {}".format(meanPGS, semPGS, st.norm.interval(alpha=0.95, loc=meanPGS, scale=semPGS)))
    
    meanPGSR = np.mean(PGSR)                                      
    semPGSR = st.sem(PGSR)                                      
    print("Mean Percentage of Global Shortage - Modified Demand", meanPGSR)
    f.write ("\n Mean Percentage of Global Shortage - Modified Demand: {} semPGSR: {} CIPGSR: {}".format(meanPGSR, semPGSR, st.norm.interval(alpha=0.95, loc=meanPGSR, scale=semPGSR)))
    
    semS, CIS, semPS, CIPS = {}, {}, {}, {}  
    SEA, PSEA = {}, {}                                
    
    PSREA, semPSR, CIPSR = {}, {}, {}
    
    for k in K:
        SEA[k] = np.zeros(numE)
        PSEA[k] = np.zeros(numE)
        
        PSREA[k] = np.zeros(numE)
        
        for w in range(numE):
            SEA[k][w] = SE[k,w]
            PSEA[k][w] = PSE[k,w]
            PSREA[k][w] = PSRE[k,w]
            
        meanS[k] = np.mean(SEA[k])
        semS[k] = st.sem(SEA[k])
        
        meanPS[k] = np.mean(PSEA[k])
        semPS[k] = st.sem(PSEA[k]) 
        
        CIS[k] = st.norm.interval(alpha=0.95, loc=meanS[k], scale=semS[k])
        CIPS[k] = st.norm.interval(alpha=0.95, loc=meanPS[k], scale=semPS[k])
        
        meanPSR[k] = np.mean(PSREA[k])
        semPSR[k] = st.sem(PSREA[k]) 
        
        CIPSR[k] = st.norm.interval(alpha=0.95, loc=meanPSR[k], scale=semPSR[k])
    

    shortage = pd.DataFrame(columns=["ISO","meanS","semS","CIS","meanPS","semPS","CIPS","meanPSR","semPSR","CIPSR"])
    for k in K:
        shortage = pd.concat([shortage, pd.DataFrame.from_records([{"ISO":k,"meanS":meanS[k],"semS":semS[k],"CIS":CIS[k],
                                             "meanPS":meanPS[k],"semPS":semPS[k],"CIPS":CIPS[k],"meanPSR":meanPSR[k],"semPSR":semPSR[k],"CIPSR":CIPSR[k] }])], ignore_index=True)
       
    flowV = pd.DataFrame(columns=["From","To","meanV"])
    for key in V.keys():
        if meanV[key] > 0:
            flowV = pd.concat([flowV, pd.DataFrame.from_records([{"From":key[0],"To":key[1],"meanV":meanV[key]}])], ignore_index=True)
    
    flowU = pd.DataFrame(columns=["From","To","meanU"])
    for key in U.keys():
        if meanU[key] > 0:
            flowU = pd.concat([flowU, pd.DataFrame.from_records([{"From":key[0],"To":key[1],"meanU":meanU[key]}])], ignore_index=True)
            
    Yhat = pd.DataFrame(columns=["ISO","Y"])
    for j in J:
        if Y_hat[j] > 0:
            Yhat = pd.concat([Yhat, pd.DataFrame.from_records([{"ISO":j,"Y":Y_hat[j]}])], ignore_index=True)
    
    scenarios_dg = pd.DataFrame(columns=["ISO","Scenario","xi_d","xi_dfinal","xi_Fprime","xi_F","PS"])
    for key in xi_d.keys():
        scenarios_dg = pd.concat([scenarios_dg, pd.DataFrame.from_records([{"ISO":key[0],"Scenario":key[1],"xi_d":xi_d[key],
                                            "xi_dfinal":xi_dfinal[key], "xi_Fprime":xi_Fprime[key], "xi_F":xi_F[key], "PS":PSE[key]}])], ignore_index=True)
    
    scenarios_sc = pd.DataFrame(columns=["ISO","Scenario","xi_sc"])
    for key in xi_sc.keys():
        scenarios_sc = pd.concat([scenarios_sc, pd.DataFrame.from_records([{"ISO":key[0],"Scenario":key[1],"xi_sc":xi_sc[key]}])], ignore_index=True)
    
    scenarios_pc = pd.DataFrame(columns=["ISO","Scenario","xi_pc","Y"])
    for key in xi_pc.keys():
        scenarios_pc = pd.concat([scenarios_pc, pd.DataFrame.from_records([{"ISO":key[0],"Scenario":key[1],"xi_pc":xi_pc[key],"Y":Y_hat[key[0]]}])], ignore_index=True)
    
    scenarios_co = pd.DataFrame(co)
    scenarios_gs = pd.DataFrame(GS)
    scenarios_pgs = pd.DataFrame(PGS)
    scenarios_pgsr = pd.DataFrame(PGSR)
    scenarios_gd = pd.DataFrame(GD)
    scenarios_grd = pd.DataFrame(GRD)
    
    scenarios_rmcost = pd.DataFrame(rm_cost)
    scenarios_drugcost = pd.DataFrame(drug_cost)
    scenarios_ssof = pd.DataFrame(ssof_hat)
    scenarios_csincome = pd.DataFrame(cs_income)
    scenarios_coincome = pd.DataFrame(co_income)
    scenarios_salescost = pd.DataFrame(sales_cost)
    scenarios_salesvol = pd.DataFrame(sales_vol)
    
    mean_ssof = np.mean(ssof_hat)
    mean_sales_cost = np.mean(sales_cost)
    mean_co_income = np.mean(co_income)
    mean_cs_income = np.mean(cs_income)
    mean_rm_cost = np.mean(rm_cost)
    mean_drug_cost = np.mean(drug_cost)
    mean_sales_vol = np.median(sales_vol)
    
    f.write("\n mean ssof:{}".format(mean_ssof))
    f.write("\n mean sales cost:{}".format(mean_sales_cost))
    f.write("\n mean sales vol:{}".format(mean_sales_vol))
    f.write("\n mean co income:{}".format(mean_co_income))
    f.write("\n mean cs income:{}".format(mean_cs_income))
    f.write("\n mean rm cost:{}".format(mean_rm_cost))
    f.write("\n mean drug cost:{}".format(mean_drug_cost))
    
    writer = pd.ExcelWriter("Results.xlsx")
    shortage.to_excel(writer,"S")
    flowV.to_excel(writer,"V")
    flowU.to_excel(writer,"U")
    Yhat.to_excel(writer,"Y")
    scenarios_dg.to_excel(writer,"xi_d_g")
    scenarios_sc.to_excel(writer,"xi_sc")
    scenarios_pc.to_excel(writer,"xi_pc")
    scenarios_co.to_excel(writer,"Co")
    scenarios_gs.to_excel(writer,"GS")
    scenarios_pgs.to_excel(writer,"PGS")
    scenarios_gd.to_excel(writer,"GD")
    scenarios_grd.to_excel(writer,"GRD")
    scenarios_pgs.to_excel(writer,"PGSR")
    
    scenarios_rmcost.to_excel(writer,"rm_cost")
    scenarios_drugcost.to_excel(writer,"drug_cost")
    scenarios_csincome.to_excel(writer,"cs_income")
    scenarios_coincome.to_excel(writer,"co_income")
    scenarios_salescost.to_excel(writer,"sales_cost")
    scenarios_ssof.to_excel(writer,"ssof_cost")
    scenarios_salesvol.to_excel(writer,"sales_vol")
    
    writer.save()
    writer.close()
    
    end_timeD = (time.time()-ini_timeD)     # time of getting data and statistics
    end_time = (time.time()-ini_time)       # execution time of the whole code without upload of packages
    print('Execution Time',end_time)
    f.write("\n General Execution Time:{}".format(end_time))
    f.write("\n Data Time:{}".format(end_timeD))
    f.close()                               # close aggregate txt file
    
    f = open("Yr.txt","a")   
    f.write("ObjRep_ObjN_Y: {}".format(objvals))
    f.close()
    