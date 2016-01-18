import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

################
#Pars
################
dat_types = {"SigCaught" : np.float32, "NSigs" : np.uint8, "NMissedSigs" : np.uint8, "NDecMade" : np.uint8, "CrT" : np.uint16, "NTp" : np.uint8, "NBr" : np.uint8, "MaxLev" : np.uint8, "NEmps" : np.uint8, "-1" : np.int16,
             "ImpVar" : np.float32, "NoiseDet" : np.float32, "CatchT" : np.uint16, "ActT" : np.uint16, "DirT" : np.uint16, "DecT" : np.uint16, "Sigma1" : np.float32, "Sigma2" : np.float32, "CrOcc" : np.float32}

#pars_pres = ["SigCaught", "NSigs", "NMissedSigs", "CrT", "NTp", "NBr", "MaxLev", "NEmps", "-1"]
pars_pres = ["SigCaught", "NSigs", "CrT", "MaxLev", "NEmps", "-1"]

min_emp = 5
max_emp = 13

################
#Functions
################
def getFileNames(dir_name):
    return [f for f in listdir(dir_name) if (isfile(dir_name + f) and f[:7] == "Results")]

def decodePar(s):
    if s == "Fl" or s == "Sm" or s == "Mi":
        return 1
    elif s == "Hi" or s == "Su" or s == "Me":
        return 2
    return 3

def processFile(dir_name, f):
    x1 = pd.read_csv(dir_name + f, sep = ";", dtype = dat_types)
    x1 = x1[pars_pres]

    pars_add = f.split('=')
    x1["OrgSt"] = decodePar(pars_add[1][:2])
    x1["Cr"] = decodePar(pars_add[2][:2])
    x1["Mode"] = decodePar(pars_add[3][:2])
    return x1

################
#Calculations
################
dir_name = "C:\\PyCharm Community Edition 5.0.3\\Projects\\CrisisModeling\\"
files = getFileNames(dir_name)
x_flat = pd.DataFrame()
x_hier = pd.DataFrame()

for f in files:
    print(f)
    x1 = processFile(dir_name, f)
    if(x1["OrgSt"][0] == 1): #if flat
        x_flat = x_flat.append(x1.drop("OrgSt", 1), ignore_index = True)
    else: # if hierarchy
        x_hier = x_hier.append(x1.drop("OrgSt", 1), ignore_index = True)


#x_flat.reset_index()
#x_hier.reset_index()

x_hier = x_hier[(x_hier["NEmps"] >= min_emp) & (x_hier["NEmps"] <= max_emp)] #change to distribution of nemps in x_flat

print(x_flat.describe())
print(x_hier.describe())

#x_flat.to_csv(dir_name + "0Results.Flat.csv")
#x_hier.to_csv(dir_name + "0Results.Hierarchy.csv")

