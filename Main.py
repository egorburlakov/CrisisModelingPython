import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

################
#Pars
################
dat_types = {"SigCaught" : np.float32, "NSigs" : np.uint8, "NMissedSigs" : np.uint8, "NDecMade" : np.uint8, "CrT" : np.uint16, "NTp" : np.uint8, "NBr" : np.uint8, "MaxLev" : np.uint8, "NEmps" : np.uint8, "-1" : np.int16,
             "ImpVar" : np.float32, "NoiseDet" : np.float32, "CatchT" : np.uint16, "ActT" : np.uint16, "DirT" : np.uint16, "DecT" : np.uint16, "Sigma1" : np.float32, "Sigma2" : np.float32, "CrOcc" : np.float32}

proc_dtypes = {"SigCaught" : np.float32, "NSigs" : np.uint8, "CrT" : np.uint16, "NEmps" : np.uint8, "TpT" : np.int16, "Mode" : np.int8, "TpLoad" : np.float32, "Cr" : np.int8}

#pars_pres = ["SigCaught", "NSigs", "NMissedSigs", "CrT", "NTp", "NBr", "MaxLev", "NEmps", "-1"]
pars_pres = ["SigCaught", "NSigs", "CrT", "NEmps", "-1"]

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
    x1 = pd.read_csv(dir_name + f, sep = ";", dtype = dat_types, usecols = pars_pres)
    x1 = x1[pd.notnull(x1['SigCaught'])]

    pars_add = f.split('=')
    x1["OrgSt"] = decodePar(pars_add[1][:2])
    #x1["Cr"] = decodePar(pars_add[2][:2])
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
    y["Drop"] = np.random.binomial(n = 1, size = y.shape[0], p = y[col + "_p"]) #choosing the raws to drop
    y = y[y.Drop == 1] #dropping the raws
    y.drop(["Drop", col + "_p", "index"], 1, inplace = True)
    return y

################
#Reading
def readInitialFiles(dir_name):
    files = getFileNames(dir_name)
    flat = pd.DataFrame()
    hier = pd.DataFrame()
    
    for f in files:
        print(f)
        x1 = processFile(dir_name, f)
        if(x1["OrgSt"][0] == 1): #if flat
            flat = flat.append(x1.drop("OrgSt", 1), ignore_index = True)
        else: # if hierarchy
            hier = hier.append(x1.drop("OrgSt", 1), ignore_index = True)
    
    #Transforming
    hier = hier[(hier["NEmps"] >= min_emp) & (hier["NEmps"] <= max_emp)] #change to distribution of nemps in flat
#    hier = dropExcessiveRaws(hier, flat, "NEmps")
    
    hier.rename(columns = {"-1" : "TpT"}, inplace = True)
    flat.rename(columns = {"-1" : "TpT"}, inplace = True)
    hier["TpLoad"] = hier["TpT"] / hier["CrT"]
    flat["TpLoad"] = flat["TpT"] / flat["CrT"]
    hier["Cr"] = np.where(hier.NSigs > 7, 2, 1)
    flat["Cr"] = np.where(flat.NSigs > 7, 2, 1)
    return flat, hier

def readProcessedFiles(dir_name):
    flat = pd.read_csv(dir_name + "0Results.Flat.csv", sep = ",", dtype = proc_dtypes)
    hier = pd.read_csv(dir_name + "0Results.Hier.csv", sep = ",", dtype = proc_dtypes)
    print("Processed Files read successfully")
    return flat, hier

dir_name = "C:\\PyCharm Community Edition 5.0.3\\Projects\\CrisisModeling\\"
all = readProcessedFiles("C:\\PyCharm Community Edition 5.0.3\\Projects\\CrisisModeling\\")
flat = all[0]
hier = all[1]

#all = readInitialFiles(dir_name)
#flat = all[0]
#hier = all[1]
#hier = hier.append(all[0], ignore_index = True)
#flat = flat.append(all[1], ignore_index = True)

#hier.to_csv(dir_name + "0Results.Hier.csv")
#flat.to_csv(dir_name + "0Results.Flat.csv")
#hier.to_csv(dir_name + "0Results.Hier.csv", mode = "a")
#flat.to_csv(dir_name + "0Results.Flat.csv", mode = "a")

#Aggregating
h_nu_nsigs = hier["SigCaught"].groupby([hier["NSigs"], hier["Mode"]]).mean().unstack()
f_nu_nsigs = flat["SigCaught"].groupby([flat["NSigs"], flat["Mode"]]).mean().unstack()
h_nu_nemps = hier["SigCaught"].groupby([hier["NEmps"], hier["Cr"], hier["Mode"]]).mean().unstack().reset_index()
f_nu_nemps = flat["SigCaught"].groupby([flat["NEmps"], flat["Cr"], flat["Mode"]]).mean().unstack().reset_index()

h_eta_nsigs = hier["TpLoad"].groupby([hier["NSigs"], hier["Mode"]]).mean().unstack()
f_eta_nsigs = flat["TpLoad"].groupby([flat["NSigs"], flat["Mode"]]).mean().unstack()
h_eta_nemps = hier["TpLoad"].groupby([hier["NEmps"], hier["Mode"]]).mean().unstack()
f_eta_nemps = flat["TpLoad"].groupby([flat["NEmps"], flat["Mode"]]).mean().unstack()

#Printing
#, ticks = [3, 5, 7, 9, 11, 13, 15, 17, 19]
fig = plt.figure(facecolor = "white")
#fig.subplots_adjust(wspace = 0, hspace = 0)
ax1 = fig.add_subplot(2, 2, 1)
plt.plot(h_nu_nsigs.index, h_nu_nsigs[2], linestyle = "--", marker = "o", color = (0, 0.5, 0), label = "Centralized", linewidth = 3)
plt.plot(f_nu_nsigs.index, f_nu_nsigs[2], linestyle = "-", marker = "^" , color = (0.5, 0, 0), label = "Decentralized", linewidth = 3)
ax1.legend(loc = "best", fontsize = 11)

plt.plot(h_nu_nsigs.index, h_nu_nsigs[1], linestyle = "--", marker = "o", color = "g", label = "Centralized")
plt.plot(h_nu_nsigs.index, h_nu_nsigs[3], linestyle = "--", marker = "o", color = "g", label = "Centralized")
plt.plot(f_nu_nsigs.index, f_nu_nsigs[1], linestyle = "-", marker = "^", color = "r", label = "Decentralized")
plt.plot(f_nu_nsigs.index, f_nu_nsigs[3], linestyle = "-", marker = "^", color = "r", label = "Decentralized")
ax1.set_ylabel(r'$\eta$', fontsize = 20)
ax1.set_xlabel("N of Signals", fontsize = 16)

ax2 = fig.add_subplot(2, 2, 2)
plt.plot(h_nu_nemps[h_nu_nemps.Cr == 1].NEmps, h_nu_nemps[h_nu_nemps.Cr == 1].ix[:,3], linestyle = "--", marker = "o", color = "g", label = "Centralized")
plt.plot(f_nu_nemps[h_nu_nemps.Cr == 1].NEmps, f_nu_nemps[h_nu_nemps.Cr == 1].ix[:,3], linestyle = "-", marker = "^", color = "g", label = "Decentralized")
ax2.legend(loc = "best", fontsize = 11)
plt.plot(h_nu_nemps[h_nu_nemps.Cr == 2].NEmps, h_nu_nemps[h_nu_nemps.Cr == 2].ix[:,3], linestyle = "--", marker = "o", color = "r", label = "Centralized")
plt.plot(f_nu_nemps[h_nu_nemps.Cr == 2].NEmps, f_nu_nemps[h_nu_nemps.Cr == 2].ix[:,3], linestyle = "-", marker = "^", color = "r", label = "Decentralized")
ax2.set_ylabel(r'$\eta$', fontsize = 20)
ax2.set_xlabel("N of Employees", fontsize = 16)

ax3 = fig.add_subplot(2, 2, 3)
plt.plot(h_eta_nsigs.index, h_eta_nsigs[2], linestyle = "--", marker = "o", color = "g", label = "Centralized")
plt.plot(f_eta_nsigs.index, f_eta_nsigs[2], linestyle = "-", marker = "^", color = "r", label = "Decentralized")
ax3.legend(loc = "best", fontsize = 11)
ax3.set_ylabel(r'$\nu$', fontsize = 20)
ax3.set_xlabel("N of Signals", fontsize = 16)


ax4 = fig.add_subplot(2, 2, 4)
plt.plot(h_eta_nsigs.index, h_eta_nsigs[1], linestyle = "--", marker = "o", color = "g", label = "Centralized", linewidth = 1)
plt.plot(h_eta_nsigs.index, h_eta_nsigs[2], linestyle = "--", marker = "o", color = (0, 0.5, 0), label = "Centralized", linewidth = 3)
plt.plot(h_eta_nsigs.index, h_eta_nsigs[3], linestyle = "--", marker = "o", color = "g", label = "Centralized", linewidth = 1)

plt.plot(f_eta_nsigs.index, f_eta_nsigs[1], linestyle = "-", marker = "^", color = "r", label = "Decentralized", linewidth = 1)
plt.plot(f_eta_nsigs.index, f_eta_nsigs[2], linestyle = "-", marker = "^", color = (0.5, 0, 0), label = "Decentralized", linewidth = 3)
plt.plot(f_eta_nsigs.index, f_eta_nsigs[3], linestyle = "-", marker = "^", color = "r", label = "Decentralized", linewidth = 1)
ax4.set_ylabel(r'$\nu$', fontsize = 20)
ax4.set_xlabel("N of Signals", fontsize = 16)


#print(flat.describe())
#print(hier.describe())

#flat.to_csv(dir_name + "0Results.Flat.csv")
#hier.to_csv(dir_name + "0Results.Hierarchy.csv")

#sys.modules[__name__].__dict__.clear()



#Dropping excessive raws in hier to make NEmps equal
# nemp_p = mutualDistCalc(hier, flat, "NEmps") / hier.NEmps.value_counts() #bernulli distribution coefficient
# nemp_p.name = "NEmps_p"
# hier = pd.merge(hier, nemp_p.reset_index(), left_on = "NEmps", right_on = "index")
# hier.drop("index", 1, inplace = True)
# hier["Drop"] = np.random.binomial(n = 1, size = hier.shape[0], p = hier.NEmps_p)
# hier = hier[hier.Drop == 1]
# hier.drop("Drop", 1, inplace = True)
