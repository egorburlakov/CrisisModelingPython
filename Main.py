import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

################
#Pars
################
dat_types = {"SigCaught" : np.float32, "NSigs" : np.uint8, "NMissedSigs" : np.uint8, "NDecMade" : np.uint8, "CrT" : np.uint16, "NTp" : np.uint8, "NBr" : np.uint8, "MaxLev" : np.uint8, "NEmps" : np.uint8, "-1" : np.int16,
             "ImpVar" : np.float32, "NoiseDet" : np.float32, "CatchT" : np.uint16, "ActT" : np.uint16, "DirT" : np.uint16, "DecT" : np.uint16, "Sigma1" : np.float32, "Sigma2" : np.float32, "CrOcc" : np.float32}

pars_pres = ["SigCaught", "NSigs", "NMissedSigs", "CrT", "NTp", "NBr", "MaxLev", "NEmps", "-1"]
#pars_pres = ["SigCaught", "NSigs", "CrT", "MaxLev", "NEmps", "-1"]

min_emp = 5
max_emp = 13

np.random.seed(1)

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
    x1 = x1[pd.notnull(x1['SigCaught'])]
    #x1 = x1[np.isfinite(x1['SigCaught'])]

    pars_add = f.split('=')
    x1["OrgSt"] = decodePar(pars_add[1][:2])
    x1["Cr"] = decodePar(pars_add[2][:2])
    x1["Mode"] = decodePar(pars_add[3][:2])
    return x1

def mutualDistCalc(y, x0, col): #finds distribution for y[col] from series x0[col]
    n = y[col].value_counts()
    p0 = x0[col].value_counts() / x0[col].value_counts().sum()
    dist = min(n / p0) * p0
    return dist.astype(int)

def dropExcessiveRaws(y, x0, col): #drop from y raws to get the same distribution for col as in x0
    col_p = mutualDistCalc(y, x0, col) / y[col].value_counts() #bernulli distribution coefficient, i. e. probabilities of occurence for values in col
    col_p.name = col + "_p" #just for joining
    y = pd.merge(y, col_p.reset_index(), left_on = col, right_on = "index") #join probabilities to the target dataframe y
    y.drop("index", 1, inplace = True)
    y["Drop"] = np.random.binomial(n = 1, size = y.shape[0], p = y[col + "_p"]) #choosing the raws to drop
    y = y[y.Drop == 1] #dropping the raws
    y.drop(["Drop", col + "_p"], 1, inplace = True)
    return y

################
#Reading
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

#Transforming
x_hier.rename(columns = {"-1" : "TpT"}, inplace = True)
x_flat.rename(columns = {"-1" : "TpT"}, inplace = True)

x_hier = x_hier[(x_hier["NEmps"] >= min_emp) & (x_hier["NEmps"] <= max_emp)] #change to distribution of nemps in x_flat
x_hier = dropExcessiveRaws(x_hier, x_flat, "NEmps")

#Aggregating
#SigCaught ~ NSigs
h_nu_nsigs = x_hier["SigCaught"].groupby([x_hier["NSigs"], x_hier["Mode"]]).mean()
f_nu_nsigs = x_flat["SigCaught"].groupby([x_flat["NSigs"], x_flat["Mode"]]).mean()
h_nu_nemps = x_hier["SigCaught"].groupby([x_hier["NEmps"], x_hier["Mode"]]).mean()
f_nu_nemps = x_flat["SigCaught"].groupby([x_flat["NEmps"], x_flat["Mode"]]).mean()

h_eta_nsigs = x_hier["TpT"].groupby([x_hier["NSigs"], x_hier["Mode"]]).mean()
f_eta_nsigs = x_flat["TpT"].groupby([x_flat["NSigs"], x_flat["Mode"]]).mean()
h_eta_nemps = x_hier["TpT"].groupby([x_hier["NEmps"], x_hier["Mode"]]).mean()
f_eta_nemps = x_flat["TpT"].groupby([x_flat["NEmps"], x_flat["Mode"]]).mean()

#print(x_flat.describe())
print(x_hier.describe())

#x_flat.to_csv(dir_name + "0Results.Flat.csv")
#x_hier.to_csv(dir_name + "0Results.Hierarchy.csv")

#sys.modules[__name__].__dict__.clear()



#Dropping excessive raws in x_hier to make NEmps equal
# nemp_p = mutualDistCalc(x_hier, x_flat, "NEmps") / x_hier.NEmps.value_counts() #bernulli distribution coefficient
# nemp_p.name = "NEmps_p"
# x_hier = pd.merge(x_hier, nemp_p.reset_index(), left_on = "NEmps", right_on = "index")
# x_hier.drop("index", 1, inplace = True)
# x_hier["Drop"] = np.random.binomial(n = 1, size = x_hier.shape[0], p = x_hier.NEmps_p)
# x_hier = x_hier[x_hier.Drop == 1]
# x_hier.drop("Drop", 1, inplace = True)
