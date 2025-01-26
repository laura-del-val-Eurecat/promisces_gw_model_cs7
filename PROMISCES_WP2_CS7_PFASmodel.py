# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:11:42 2025

@author: laura.delval
"""



import pandas as pd
import phydrus as ps
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
import flopy as fp
import shutil



plt.ioff()


##############################################################################
############################### HYDRUS 1D ####################################
##############################################################################



# 0. BASIC INFORMATION

## 0.1 SET DIRECTORY AND OPEN FILES

# Folder for Hydrus files to be stored
ws = "H1D_CS7"
# Path to folder containing hydrus.exe
exe = 'hydrus.exe'
# Description/Model name
desc = "H1D_CS7" 


# 1. CREATE HYDRUS1D  MODEL

## 1.1 Create model
ml = ps.Model(exe_name=exe, ws_name=ws, name="model", description=desc,
              mass_units="mmol", time_unit="days", length_unit="cm")


## 1.2 Temporal discretization
tiempo_estacionario = 1201 # Time to stationary state as initial condition for transient simulation (Days)
Time_transient = 2768 # Days
tiempo_total = tiempo_estacionario + Time_transient # Total simmulated time
printtimes = ([1, 1200, 1230, 1260, 1290, 1320, 1350, 1380, 1410, 1440, 1500, 1530,  1560, 1600, 1700, 1800, 1900, 2000, 2500, 3000, 3500, 3850])

times = ml.add_time_info(tmax=tiempo_total,  # Final time of the simulation [T]
                         print_times= True, # Set to True. if information of pressure head, water contents, temperatures, and concentrations in observation nodes, is to be printed at a constant                        
                         dt=0.001,  # Initial time increment [T].
                         dtmin=0.00001,  # Minimum permitted time increment [T].
                         dtmax=0.1)  # maximum permitted time increment [T]


## 1.3 Water/Solutes inflow parameters
ml.add_waterflow(linitw=False,  # True - initial condition is given in terms of the water content. False - pressure head
                 top_bc=2,  # Upper BC: 1 = Constant Flux.--> Recharge / 2 = Atmospheric Boundary Condition with Surface Layer
                 bot_bc=4)  # Lower Boundary Condition:4 = Free Drainage.


ml.add_solute_transport(tpulse=tiempo_total, # Time duration of the concentration pulse [T].
                        ctola = 0.00000001, # Absolute concentration tolerance [ML-3]
                        ctolr = 0.00000001, # Relative concentration tolerance [-]
                        maxit=30) # Maximum iterations allowed during any time step for solute transport 



## 1.4 Atmospheric Boundary condition


# Atmospheric data (Precipitacion and ET potencial --> estimated with Hargreaves)
atm = pd.read_csv("H1D_Input_Atmos.csv", decimal=",", sep=";")


# Frequency of pollution events
Frequency = 15 # days

# Estimate pollution event concentration:
Vol_apagar_coche = 20000 # l. = mm/m2
Area_Infiltracion = 500 # m2
vol_m2 = Vol_apagar_coche/Area_Infiltracion # l/m2 = mm 
Vol_events = (vol_m2 * 3 )/10 

# Imput Concentrations ug/l)
Diluted_C_PFOS  =  50  

# Molecular weigth  mg/mmol: 
PM_PFOS =   500.13
# PM_PFOA =   414.07
# PM_PFHxS =  400.11
# PM_PFBA =   214.04


# Generate recharge array with polluting events:
concIn = round(Diluted_C_PFOS/(1000 * PM_PFOS *1000), 10) # mmol/cm3 PFOSin each polluting event

flag =True
counter = 0
for p in range(tiempo_total):
    
    if p>tiempo_estacionario:
        
        if counter == Frequency:
            
            if atm['Prec'][p]>0:
                atm['Prec'][p] = atm['Prec'][p] + Vol_events
                atm['cTop'][p] = (concIn * Vol_events) / (atm['Prec'][p] + Vol_events) 
                counter = 0
                
            else:
                atm['Prec'][p] = atm['Prec'][p] + Vol_events
                atm['cTop'][p] = concIn
                counter = 0
            
        else:
            counter = counter + 1
            atm['cTop'][p] = 0.0
    else:
        atm['cTop'][p] = 0.0
        
        
# Join stationary and transient time series:
ml.add_atmospheric_bc(atm[:tiempo_total])


## 1.5 Define layers and materials
m = ml.get_empty_material_df(n=9) 

# thr - θr (residual water content, cm3/cm3)
# ths - θs (saturated water content, cm3/cm3)
# Alpha (the hydraulic shape parameter α, 1/cm)
# n (the hydraulic parameter n, dimensionless)
# Ks (the saturated hydraulic conductivity, cm/day)
# l (Mualem’s pore connectivity exponent; the default value of 0.5 is used here)
# bulk.d (Bulk density, ρ [ML-3])
# DisperL (Longitudinal dispersivity, DL  [L])
# frac (Dimensionless fraction of adsorption sites classified as type-1 sites) i.e., sites with instantaneous sorption when the chemical nonequilibrium option is considered [-] --> Frac. = 0 (all sorption sites are kinetic – as in Figure 6c.1; no equilibrium  sorption)
# mobile_wc (The immobile water content. Set equal to 0 when the physical nonequilibrium option is not considered.)

# Hydraulic conductivity (cm/day)
Ks = [   20000,      200,       20,      200,    20000,      200,       20,      200,    20000]
  

#                            columns=[  "thr",  "ths", "Alfa",      "n",      "Ks",     "l", "bulk.d", "DisperL", "frac", "mobile_wc"])
m.loc[[1, 2, 3, 4, 5, 6, 7, 8, 9]] = [[0.0426, 0.3846, 0.0349,   1.4271,     Ks[0],     0.5,      1.5,        3,      1,     0 ],     # Relleno antrópico (1.6m)
                                      [0.0426, 0.3846, 0.0349,   1.4271,     Ks[1],     0.5,      1.5,        3,      1,     0 ],      # --> Capa amortiguacion (0.1 m)
                                      [0.0387,  0.387, 0.0267,   1.4484,     Ks[2],     0.5,     1.29,        3,      1,     0 ],     # Arenas finas limosas (2m)  
                                      [0.0387,  0.387, 0.0267,   1.4484,     Ks[3],     0.5,     1.29,        3,      1,     0 ],      # --> Capa amortiguacion (0.1 m)
                                      [0.0387,  0.387, 0.0267,   1.4484,     Ks[4],     0.5,      1.5,        3,      1,     0 ],     # Gravas redondeadas (1.2m)
                                      [0.0387,  0.387, 0.0267,   1.4484,     Ks[5],     0.5,      1.5,        3,      1,     0 ],       # --> Capa amortiguacion (0.1 m)
                                      [0.0438, 0.3873, 0.0332,    1.492,     Ks[6],     0.5,     1.49,        3,      1,     0 ],     # Arenas finas limosas (0.3m)
                                      [0.0438, 0.3873, 0.0332,    1.492,     Ks[7],     0.5,     1.49,        3,      1,     0 ],       # --> Capa amortiguacion (0.1 m)
                                      [0.0387,  0.387, 0.0267,   1.4484,     Ks[8],     0.5,      1.5,        3,      1,     0 ]]     # Gravas redondeadas (0.9m)

ml.add_material(m)



## 1.6 Add profile information
profile = ps.create_profile(bot=[-160,-170, -360,-370, -480,-490, -510,-520, -600],  # Bottom of the soil column. If a list is provided, multiple layers are created and other arguments need to be of the same length (e.g. mat).
                            h=-600,  # Initial values of the pressure head.
                            dx=2,  # Size of each grid cell. cm
                            lay = m.index, # The subregions may or may not coincide with the material regions. Each subregion is characterized by an integer code that runs from 1 to NLay (the total number of subregions). A subregion code is assigned to each element in the flow domain
                            mat=m.index,  # Material number (for heterogeneity)
                            sconc=0,
                            conc = 0)
                            

ml.add_profile(profile)


## 1.7 Add observation nodes:
    
# List of floats denoting the depth of the nodes.
ml.add_obs_nodes([-50, -100, -160, -270, -360, -480, -500, -510, -590])


## 1.8  Add solute and transport parameters:
    
# Kd Adsorption isotherm coefficient (sol1["ks"] ) --> [M-1L3] cm3/g if bulk density in g/cm3
KDs = [  0.001,     0.03,      0.6,     0.03,    0.001,     0.03,      0.3,     0.03,    0.001] 

sol1 = ml.get_empty_solute_df()  

# Adsorcion lineal:
sol1["beta"] = 1 # no units
sol1["nu"] = 0.0 # M-1L3
sol1["ks"] = KDs  # Kd Adsorption isotherm coefficient [M-1L3] [cm3/g]

ml.add_solute(sol1, difw=0, difg=0, top_conc=0)


# 2. Write HYDRUS input and run model
ml.write_input()

ml.simulate()


# 3. Save results to .xlsx for imput to MODFLOW/MT3D:
# This is the data we will use as recharge BC at the node were pollution is assumed to happen in the MF6 model
    
df_bc = ml.read_tlevel()
df_bc_c = ml.read_solutes()
df_bc_c.index = np.array(df_bc_c.index, dtype='float')


Start_date = dt.datetime.strptime('2016-1-01','%Y-%m-%d' ).date()
End_date = dt.datetime.strptime('2023-07-31','%Y-%m-%d' ).date()
periodos = (End_date - Start_date).days # days


df_MODFLOW_rech = pd.DataFrame()
df_MODFLOW_rech['cBot'] = (df_bc_c['cBot'] [tiempo_estacionario:tiempo_total])
df_MODFLOW_rech['vBot'] = df_bc['vBot'][tiempo_estacionario:tiempo_total]


dateList = []
for x in range (0, periodos):
    dateList.append(Start_date + dt.timedelta(days = x))
    
df_MODFLOW_rech.index=dateList

df_MODFLOW_rech.to_excel('H1D_Output_GWRecharge.xlsx')








##############################################################################
################################## MF6 #######################################
##############################################################################



 
# 1. MODEL VARIABLES AND BASIC INFORMATION ###############################

## 1.0. Import files:

### 1.0.1 Recharge


BC_BCTop = df_MODFLOW_rech

BC_BCTop_sinPFOS = pd.read_excel('H1D_Output_GWRecharge_sinPFOS.xlsx')
BC_BCTop['vBot_sinPFOS'] = np.array(BC_BCTop_sinPFOS.vBot)


### 1.0.2. BC Upflow
BC_Hrelative = pd.read_excel('MF6_Input_Heads.xlsx')
BC_Hrelative.index = BC_Hrelative.Date
BC_Hrelative.index = pd.to_datetime(BC_Hrelative.index).date


## 1.1. Directorio:
    
modelname = 'MF6_CS7'
modelws = './MF6_CS7'

path = os.getcwd() # path to current working directory
plt.rcParams['figure.figsize'] =(10,5) #default figure size
plt.rcParams['figure.autolayout'] = True # smae as tight_layout 



## 1.2. Grid properties:
    
Lx = 640. # 640 m (inlcuyendo 180 metros agua arriba del MW01)
Lz = 14. # m dejamos una capa de granito abajo espesa para ponerle k muy baja
Ly = 1. # 1 m de espesor del perfil

delc = 4
delv = 0.25
delr = 1.

nlay = int(np.round (Lz / delv))
ncol = int(np.round (Lx / delc ))
nrow = 1



## 1.3. Geometry

### 1.3.1. Distribucion materiales

materiales = np.zeros((nlay, nrow, ncol), dtype=int)
centroids_x = np.linspace(delc/2,Lx-(delc/2),ncol)
centroids_y = np.linspace(Lz-(delv/2),delv/2,nlay)

DB_layers = pd.read_excel("MF6_Input_Geometry.xlsx")
DB_layers.index = DB_layers.X_model
for l in range(len(DB_layers.columns[2:])):
    locals()[DB_layers.columns[2:][l]]=DB_layers[DB_layers.columns[2:][l]]
    locals()[DB_layers.columns[2:][l]+'_modelo'] = pd.DataFrame(index = centroids_x)
    dummy = pd.concat([locals()[DB_layers.columns[2:][l]+'_modelo'],locals()[DB_layers.columns[2:][l]]])
    dummy = dummy.sort_index()
    locals()['L'+str(l+1)] = dummy.interpolate()
    #drop duplicates from index
    print ('L'+str(l+1), DB_layers.columns[2:][l])
    
    for c in range(ncol):
        dummy = locals()['L'+str(l+1)][locals()['L'+str(l+1)].index== centroids_x [c]].iloc[0][0]
        dummy2 = np.where(centroids_y[centroids_y<dummy])
        materiales[-(np.array(dummy2, dtype=int)+1),:,c] = l+1

# Generamos dominio del modelo con celdas activas e inactivas:
ibound = np.zeros_like(materiales)
ibound[np.where(materiales >0)] = 1
        
# plot materials distribution:  

fig = plt.figure(num= 1)
materiales_names = np.array(DB_layers.columns[3:], dtype=str)    
plt.matshow(materiales[:,0,:])



# botm (double) is the bottom elevation for each cell.
bottom = [Lz - delv * k for k in range(1, nlay + 1)] # Bottom of each cell in the domain
bottom_mf = np.zeros_like(materiales, dtype=float)
for i in range(nlay):
    bottom_mf[i,:,:] = bottom[i]

# top (double) is the top elevation for each cell in the top model layer.
top_lay = [Lz  for k in range(0, ncol )]




## 1.4. Aquifer properties

### 1.4.1. Conductividad Hidraulica:
    
materials =  3  # 1) gravels, 2) weathered granite, 3) granite

k_hor_n = [1., 1., 0.1]
k_hor = [k_hor_n[0], k_hor_n[1], k_hor_n[2]] # horizontal hydraulic condctivity, m/d
k_ver = [k_hor[0]/2, k_hor[1]/2, k_hor[2]/2]  # vertical hydraulic conductivity, m/d




Kh_domain = np.array(materiales, dtype=float)
for i in range(materials):
    Kh_domain[Kh_domain==i+1] = k_hor[i]
    
Kv_domain = np.array(materiales, dtype=float)
for i in range(materials):
    Kv_domain[Kv_domain==i+1] = k_ver[i]



### 1.4.2. Porosity:
    
porosity = [0.2, 0.2, 0.2] # porosity of all layers, -

Por_domain = np.array(materiales, dtype=float)
for i in range(materials):
    Por_domain[Por_domain==i+1] = porosity[i]



## 1.5. Boundary conditions

Start_date = dt.datetime.strptime('2016-1-01','%Y-%m-%d' ).date()
End_date = dt.datetime.strptime('2021-11-16','%Y-%m-%d' ).date()
periodos = (End_date - Start_date).days # days

### 1.5.1. Recharge trough top layer

#### 1.5.1.1 Effective recharge

# Mean anual effective recharge:
recharge = 5.69e-4 # m/d (0.57 mm/d) # Recarga efectiva media diaria (Datos P y T Fogars de la Selva, y Irradiacion de Puig Sesolles )
recharge = (BC_BCTop['vBot_sinPFOS'].mean()) * -1. / 100 # de cm/day a m/day

# Recharge time series for period:
PM_PFOS =   500.13 # mg/mmol: 
BC_BCTop_period = BC_BCTop[(BC_BCTop.index>Start_date)&(BC_BCTop.index<End_date)]
BC_BCTop_period['vBot_m'] = (BC_BCTop_period.vBot * -1. )/100 # de cm/day a m/day
BC_BCTop_period['cBot_g'] = BC_BCTop_period.cBot * 1e6 * PM_PFOS / 1000 # de mmol/cm3 a g/m3
BC_BCTop_period['vBot_sinPFOS_m'] = (BC_BCTop_period.vBot_sinPFOS * -1. )/100 # de cm/day a m/day


# Add Pollution events
X_pollution = int(np.round (284. / delc )) # column
Z_pollution = int(np.round ((Lz - 12. )/ delv )) # layer


# build array recharge data per stress period
# stress_period_data ([cellid, recharge, aux, boundname]) –
# cellid = For a structured grid that uses the DIS input file, CELLID is the layer, row, and column
# recharge = recharge flux rate This rate is multiplied inside the program by the surface area of the cell to calculate the volumetric recharge rate.

rch_recarray ={}

rech_periodos = np.concatenate((np.array([recharge]), np.array(BC_BCTop_period['vBot_m'])))
rech_periodos_sinPFOS = np.concatenate((np.array([recharge]), np.array(BC_BCTop_period['vBot_sinPFOS_m'])))
conc_periodos = np.concatenate((np.array([0.0]), np.array(BC_BCTop_period['cBot_g'])))

for p in range(periodos):
    print (str(p))
    dummy =  list()
    
    for c in range(1,ncol-1):
        flag =True
        
        for l in range(nlay):
            if flag:
                if ibound[l,0,c]==1:
                    
                    flag = False
                    if c == X_pollution: 
                        dummy.append(((l,0,c),rech_periodos[p], conc_periodos[p]))
                    else:
                        dummy.append(((l,0,c),rech_periodos_sinPFOS[p], 0.0))

    rch_recarray.update({p:dummy})


Time_recharge =([])
R_recharge =([])
R_pollution = ([])
Conc_recharge =([])
Days = len(rch_recarray)
for d in range(Days):
    Time_recharge.append(d)
    R_recharge.append(rch_recarray[d][0][1]) # recharge flux
    R_pollution.append(rch_recarray[d][X_pollution-1][1]) # contaminant input


fig = plt.figure(num=2)

plt.plot(Time_recharge,R_recharge, c='b', label = 'Recharge through top layer' )
plt.legend(loc=2)
plt.ylabel (' Recharge m/d)')
plt.twinx()
plt.plot(Time_recharge,R_pollution, c='r', label = 'Recharge at pollution source' )
plt.legend(loc=1)
plt.ylabel ('Pollution flux m/d')


### 1.5.2. River/drain BC condition
# build array per stress period:

lrcec ={}
loc_BC_drain = sum(ibound[:,0,-1]==0)

for p in range(len (rech_periodos)):
        lrcec.update({p: [loc_BC_drain , 0, ncol-1, L1.iloc[-1][0], 100.]})
    
    

### 1.5.3. Upstream BC ( Head)

# Head time series for period:
BC_H_period = BC_Hrelative[(BC_Hrelative.index>Start_date)&(BC_Hrelative.index<End_date)]
BC_H_period['H'] = BC_H_period.H_rel + Lz
chd_spd ={}

H_periodos = np.concatenate((np.array([Lz]), np.array(BC_H_period['H'])))
H_conc_periodos = np.concatenate((np.array([0.0]), np.zeros(len (H_periodos)-1)))

for p in range(len (H_periodos)):
    dummy =  list()
    
    for i in range(nlay):
          dummy.append(((i, 0, 0), H_periodos[p], H_conc_periodos[p] )) # [(layer, row, column), head, aux ]
          chd_spd.update({p:dummy})
      


### 1.5.4. Recharge Constant concentration BC 
# stress_period_data : [cellid, conc, aux, boundname]
# CELLID is the layer, row, and column
# conc (double) is the constant concentration value. 
# aux (double) represents the values of the auxiliary variables for each constant concentration. 
# boundname (string) name of the constant concentration cell.

cncspd ={}
for p in range(len (rech_periodos)):
    
    if p == 0:
        cncspd.update({p: [[(Z_pollution, 0, X_pollution), 0.0]]})

    else:
        cncspd.update({p: [[(Z_pollution, 0, X_pollution), conc_periodos[p]]]})
        


## 1.6. Initial conditions

### 1.6.1. Initial head conditions based on top boundary (fully saturated)
ic_heads = np.zeros_like(materiales, dtype=float)
for c in range(np.shape(ic_heads)[2]):
    for r in range(np.shape(ic_heads)[0]):
        if materiales[r,0,c]>0:
            ic_heads[r,0,c] = L1.iloc[c][0]
ic_heads[ic_heads==0.0]=-999.9 # mark no flow cells



### 1.6.1. Initial concentrations
i_conc = np.zeros_like(ibound)
i_conc[ibound ==0]= -999.9



## 1.7. Time discretization:

nper = periodos                #Number of model stress periods (the default is 1).
perlen = np.linspace(1.0, 1.0, periodos)   #An array of the stress period lengths in days, seconds or whatever

timesteps = 5
nstp =  np.concatenate((np.array([1]), np.linspace(timesteps, timesteps, periodos)))   # Array with timesteps per stress period 
tsmult = perlen = np.linspace(1.0, 1.0, periodos)     # Multiplicador, hace que la longitud del timestep sea menor al prinicipio si se le pone valores mayores que 1
# timprs =  1           #The total elapsed time at which the simulation results are saved [ej. [8 * 365, 12 * 365, 20 * 365]]

tdis_dis = []
for i in range(nper):
    tdis_dis.append((perlen[i], nstp[i], tsmult[i]))



## 1.8 Transport parameters:
al = 0.5  # Longitudinal dispersivity ($m$)
al = 0.05  # Longitudinal dispersivity ($m$)
trpt = 0.01  # Ratio of transverse to longitudinal dispersitivity
trpv = 0.01  # Ratio of vertical to longitudinal dispersitivity
ath1 = al * trpt
atv = al * trpv
tral = 0.005  # Transverse vertical dispersivity ($m$)
dmcoef0 = 1.34e-5  # Effective diffusion coefficient ($cm^2/sec$)
dmcoef = dmcoef0 *(60*60*24)/10000 # m2/day
b_dens = 1.53 # g/l
Kd_coef = 6e-7 # PFOA



## 1.9. Solver parameters:
    
nouter = 50  #  integer value defining the maximum number of outer (nonlinear) iterations –
ninner = 100 #  integer value defining the maximum number of inner (linear) iterations
hclose = 0.000001 #  real value defining the head change criterion for convergence of the outer (nonlinear) iterations
    


## 1.10. Remove old files

if os.path.exists(modelname):
    shutil.rmtree(modelname)





# 2. CREATE MODFLOW6 MODEL  ######################################

## 2.1. Create simulation

sim = fp.mf6.MFSimulation(sim_name = modelname,
                          version = 'mf6', 
                          exe_name = path + '/mf6', # wehre MODFLOW is located in computer
                          sim_ws = modelws, #subdirectory for MODLFOW files
                          )


## 2.2. Time discretization (Tdis)
# --> This is a steady state model

tdis = fp.mf6.ModflowTdis(sim,
                          time_units ='DAYS',
                          perioddata =tdis_dis,
                          nper = nper,
                          )



## 2.3. Interative model solution (Ims)
#--> We need to tell it how to solve

ims = fp.mf6.ModflowIms(sim,
                        complexity = 'SIMPLE',
                        print_option = 'ALL',  # Controls printing of convergence information from the solver.NONE, SUMMARY, ALL
                        csv_output_filerecord = 'ALL' , # Ascii comma separated values output file to write solver convergence information. NONE, SUMMARY, ALL
                        # outer_hclose =  #  real value defining the head change criterion for convergence of the outer (nonlinear) iterations
                        outer_dvclose = hclose ,# real value defining the dependent-variable (for example, head) change criterion for convergence of the outer (nonlinear) iterations, i
                        outer_maximum = nouter, #  integer value defining the maximum number of outer (nonlinear) iterations –
                        inner_maximum = ninner ,#  integer value defining the maximum number of inner (linear) iterations
                        inner_dvclose = hclose # real value defining the dependent-variable (for example, head) change criterion for convergence of the inner (linear) iterations, in units of the dependent-variable (for example, length for head). 
                        )



## 2.4. Groundwater flow model (Gwf)

gwf = fp.mf6.ModflowGwf(sim,
                        modelname = modelname,
                        model_nam_file=f"{modelname}.nam",
                        save_flows = True,
                        )


# 3. ADD PACKAGES TO GROUNDWATER FLOW MODEL ##########################

## 3.1 Discretization package (Gwfdis)
print ('3.1 Discretization package (Gwfdis)')

dis = fp.mf6.ModflowGwfdis(gwf,
                           length_units ='METERS',
                           nlay=nlay,
                           nrow=nrow,
                           ncol=ncol,
                           delr=delr,
                           delc=delc,
                           botm=bottom_mf, #  bottom elevations of all cells
                           top=top_lay,  #top elevations of cells in topmost layer
                           idomain  = ibound
                           )


## 3.2. Node property flow package (Gwfnpf) 
#--> Define the hidrualic properties of cells

print ('3.2. Node property flow package (Gwfnpf) ')

npf = fp.mf6.ModflowGwfnpf(gwf,
                           save_flows=True,
                           save_specific_discharge = True,
                           icelltype = 0, 
                           k=Kh_domain, # horizontal k value
                           k33=Kv_domain, # vertical k value
                           )



# 3.3. Storage package (Gwfsto) 
print ('3.3. Storage package (Gwfsto) ')
sto = fp.mf6.ModflowGwfsto(gwf, ss=0, sy=0)



## 3.4 Initial condictions (Gwfic)
print ('3.4 Initial condictions (Gwfic)')
#They must be specified for all simulations, even steady simulations

ic = fp.mf6.ModflowGwfic(gwf,
                          strt=ic_heads
                          )


## 3.5. BOUNDARY CONDITIONS:
print ('3.5. BOUNDARY CONDITIONS:')
    
### 3.5.1 RIVER BC --> Drain package in this case

drn = fp.mf6.ModflowGwfdrn(gwf,
                           stress_period_data=lrcec,
                           pname='drn' #packeg name in budget
                           )



### 3.5.2. Recharge package (Gwfrcha)

rch = fp.mf6.ModflowGwfrch( gwf,
                            fixed_cell=True,
                            print_input=True,
                            maxbound=2,
                            stress_period_data = rch_recarray,
                            auxiliary="CONCENTRATION",
                            pname="RCH-1",
                            )



### 3.5.3. Specified head package (Gwfchd)

chd = fp.mf6.ModflowGwfchd( gwf,
                            stress_period_data = chd_spd,
                            auxiliary="CONCENTRATION",
                            pname='chd' #packeg name in budget,
                            )



## 3.6 Output control (Gwfoc):
    
oc = fp.mf6.ModflowGwfoc(gwf,
                         budget_filerecord=f"{modelname}.cbc",
                         head_filerecord=f"{modelname}.hds",
                         saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
                         printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
                         )
    
# 3.7 Añadir puntos de observacion NIVELES:

puntos_obs =[(Z_pollution  ,   0, X_pollution),
              (Z_pollution+4,  0, X_pollution),
              (Z_pollution+8,  0, X_pollution),
              (Z_pollution+12, 0, X_pollution),
              (Z_pollution+16, 0, X_pollution),
              (Z_pollution+20, 0, X_pollution),
              (Z_pollution+1,  0, X_pollution+1),
              (Z_pollution+5,  0, X_pollution+1),
              (Z_pollution+9,  0, X_pollution+1),
              (loc_BC_drain ,  0,        ncol-1)]



obs_dict = {
    'head.obs.csv': [('MW01_UP', 'HEAD', puntos_obs[0]),  # (nombre, tipo, (capa, fila, columna))
                     ('MW01_1M', 'HEAD', puntos_obs[1]),
                     ('MW01_2M', 'HEAD', puntos_obs[2]),
                     ('MW01_3M', 'HEAD', puntos_obs[3]),
                     ('MW01_4M', 'HEAD', puntos_obs[4]),
                     ('MW01_5M', 'HEAD', puntos_obs[5]),
                     ('MW04_1M', 'HEAD', puntos_obs[6]),
                     ('MW04_2M', 'HEAD', puntos_obs[7]),
                     ('MW04_3M', 'HEAD', puntos_obs[8]),
                     (  'RIVER', 'HEAD', puntos_obs[9])],
}

fp.mf6.ModflowUtlobs(gwf, filename = 'gwf.head.obs.csv' , continuous=obs_dict)

## 3.8 Groundwater transport model


### 3.8.1. Groundwater transport package (MFModel)

gwtname = "gwt_" + modelname

gwt = fp.mf6.MFModel(   sim,
                        model_type = "gwt6",
                        modelname = gwtname,
                        model_nam_file = f"{gwtname}.nam",
                        )
                
gwt.name_file.save_flows = True



### 3.8.2. Iterative model solution and register the gwt model with it

imsgwt = fp.mf6.ModflowIms(     sim,
                                print_option="summary",
                                complexity="complex",
                                outer_dvclose = hclose,
                                outer_maximum = nouter,
                                under_relaxation="dbd",
                                linear_acceleration="BICGSTAB",
                                under_relaxation_theta=0.7,
                                under_relaxation_kappa=0.08,
                                under_relaxation_gamma=0.05,
                                under_relaxation_momentum=0.0,
                                backtracking_number=20,
                                backtracking_tolerance=2.0,
                                backtracking_reduction_factor=0.2,
                                backtracking_residual_limit=5.0e-4,
                                inner_dvclose=hclose,
                                rcloserecord="0.0001 relative_rclose",
                                inner_maximum=ninner,
                                relaxation_factor = 1,
                                number_orthogonalizations=2,
                                preconditioner_levels=8,
                                preconditioner_drop_tolerance=0.001,
                                filename=f"{gwtname}.ims",
                                )

sim.register_ims_package(imsgwt, [gwt.name])
    
   
    
### 3.8.3. Transport discretization package

fp.mf6.ModflowGwtdis(   gwt,
                        nlay=nlay,
                        nrow=nrow,
                        ncol=ncol,
                        delr=delr,
                        delc=delc,
                        top = top_lay,
                        botm = bottom_mf,
                        idomain = ibound,
                        filename = f"{gwtname}.dis",
                        )



### 3.8.4. Transport initial concentrations

fp.mf6.ModflowGwtic(    gwt, 
                        strt = i_conc, 
                        filename = f"{gwtname}.ic")



### 3.8.5. Transport advection package

fp.mf6.ModflowGwtadv(    gwt, 
                         scheme = "UPSTREAM", #scheme used to solve the advection term (upstream, central, or TVD.)
                         filename=f"{gwtname}.adv")



### 3.8.6. Transport dispersion package

fp.mf6.ModflowGwtdsp(       gwt,
                            alh = al,       # longitudinal dispersivity in horizontal direction
                            ath1 = ath1,    # transverse dispersivity in horizontal direction.
                            atv = atv,      # transverse dispersivity when flow is in vertical direction.
                            filename = f"{gwtname}.dsp",
                            )


### 3.8.7. Transport mass storage package

fp.mf6.ModflowGwtmst(       gwt,
                            porosity = Por_domain,
                            first_order_decay=False,
                            decay=None,                 # rate coefficient for first or zero-order decay
                            decay_sorbed=None,
                            sorption = "LINEAR",        # keyword to indicate that sorption will be activated. 
                            distcoef = Kd_coef,
                            bulk_density = b_dens,          # bulk density of the aquifer in mass per length cubed
                            filename=f"{gwtname}.mst",
                            )



### 3.8.8. Transport source-sink mixing package

# sources:
sourcerecarray = [("RCH-1", "AUX", "CONCENTRATION")] #  [pname, srctype, auxname]

# pname (string) name of the flow package for which an auxiliary variable contains a source concentration. 
# srctype (string) keyword indicating how concentration will be assigned for sources and sinks.
#If the AUX  keyword is specified, then the auxiliary variable specified by the user will be assigned 
# as the concentration value for groundwater  sources (flows with a positive sign)
# auxname (string) name of the auxiliary variable in the package PNAME


fp.mf6.ModflowGwtssm(       gwt, 
                            sources = sourcerecarray, #  [pname, srctype, auxname]
                            filename = f"{gwtname}.ssm")



### 3.8.9. Transport output control package

fp.mf6.ModflowGwtoc(        gwt,
                            budget_filerecord=f"{gwtname}.cbc",
                            concentration_filerecord=f"{gwtname}.ucn",
                            concentrationprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
                            saverecord=[ ("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
                            printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
                            )



### 3.8.10. Constant concentration at upper boundary.

fp.mf6.ModflowGwtcnc(       gwt,
                            print_flows=True,
                            stress_period_data = cncspd,
                            pname = "CNC-1",
                            filename = f"{gwtname}.cnc",
                            )



### 3.8.11.  MODFLOW 6 flow-transport exchange mechanism

fp.mf6.ModflowGwfgwt(       sim,
                            exgtype="GWF6-GWT6",
                            exgmnamea = modelname,
                            exgmnameb = gwtname,
                            filename = f"{modelname}.gwfgwt",
                            )



### 3.8.12.  Observations package

puntos_obs =[(Z_pollution  ,   0, X_pollution),
              (Z_pollution+4,  0, X_pollution),
              (Z_pollution+8,  0, X_pollution),
              (Z_pollution+12, 0, X_pollution),
              (Z_pollution+16, 0, X_pollution),
              (Z_pollution+20, 0, X_pollution),
              (Z_pollution+1,  0, X_pollution+1),
              (Z_pollution+5,  0, X_pollution+1),
              (Z_pollution+9,  0, X_pollution+1),
              (loc_BC_drain ,  0,        ncol-1)]


obs_dict = {
    'conc.obs.csv': [('MW01_UP', 'CONCENTRATION', puntos_obs[0]),  # (nombre, tipo, (capa, fila, columna))
                     ('MW01_1M', 'CONCENTRATION', puntos_obs[1]),
                     ('MW01_2M', 'CONCENTRATION', puntos_obs[2]),
                     ('MW01_3M', 'CONCENTRATION', puntos_obs[3]),
                     ('MW01_4M', 'CONCENTRATION', puntos_obs[4]),
                     ('MW01_5M', 'CONCENTRATION', puntos_obs[5]),
                     ('MW04_1M', 'CONCENTRATION', puntos_obs[6]),
                     ('MW04_2M', 'CONCENTRATION', puntos_obs[7]),
                     ('MW04_3M', 'CONCENTRATION', puntos_obs[8]),
                     (  'RIVER', 'CONCENTRATION', puntos_obs[9])],
}


fp.mf6.ModflowUtlobs(gwt, filename = 'gwt.conc.obs.csv' , continuous=obs_dict)



## 3.10 Solve model

sim.write_simulation()

sim.run_simulation()














   