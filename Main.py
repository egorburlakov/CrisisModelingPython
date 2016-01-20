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
    x1 = pd.read_csv(dir_name + f, sep = ";", dtype = dat_types, usecols = pars_pres)
    x1 = x1[pd.notnull(x1['SigCaught'])]

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
x_hier["TpLoad"] = x_hier["TpT"] / x_hier["CrT"]
x_flat["TpLoad"] = x_flat["TpT"] / x_hier["CrT"]

x_hier = x_hier[(x_hier["NEmps"] >= min_emp) & (x_hier["NEmps"] <= max_emp)] #change to distribution of nemps in x_flat
x_hier = dropExcessiveRaws(x_hier, x_flat, "NEmps")

#Aggregating
h_nu_nsigs = x_hier["SigCaught"].groupby([x_hier["NSigs"], x_hier["Mode"]]).mean().unstack()
f_nu_nsigs = x_flat["SigCaught"].groupby([x_flat["NSigs"], x_flat["Mode"]]).mean().unstack()
h_nu_nemps = x_hier["SigCaught"].groupby([x_hier["NEmps"], x_hier["Cr"], x_hier["Mode"]]).mean().unstack().reset_index()
f_nu_nemps = x_flat["SigCaught"].groupby([x_flat["NEmps"], x_flat["Cr"], x_flat["Mode"]]).mean().unstack().reset_index()

h_eta_nsigs = x_hier["TpLoad"].groupby([x_hier["NSigs"], x_hier["Mode"]]).mean().unstack()
f_eta_nsigs = x_flat["TpLoad"].groupby([x_flat["NSigs"], x_flat["Mode"]]).mean().unstack()
h_eta_nemps = x_hier["TpLoad"].groupby([x_hier["NEmps"], x_hier["Mode"]]).mean().unstack()
f_eta_nemps = x_flat["TpLoad"].groupby([x_flat["NEmps"], x_flat["Mode"]]).mean().unstack()

#Printing
#, ticks = [3, 5, 7, 9, 11, 13, 15, 17, 19]
fig = plt.figure(facecolor = "white")
#fig.subplots_adjust(wspace = 0, hspace = 0)
ax1 = fig.add_subplot(2, 2, 1)
plt.plot(h_nu_nsigs.index, h_nu_nsigs[2], linestyle = "--", marker = "o", color = "g", label = "Centralized")
plt.plot(f_nu_nsigs.index, f_nu_nsigs[2], linestyle = "-", marker = "^", color = "r", label = "Decentralized")
ax1.legend(loc = "best", fontsize = 11)
ax1.set_ylabel(r'$\nu$', fontsize = 20)
ax1.set_xlabel("N of Signals", fontsize = 16)

ax2 = fig.add_subplot(2, 2, 2)
plt.plot(h_nu_nemps[h_nu_nemps.Cr == 1].NEmps, h_nu_nemps[h_nu_nemps.Cr == 1].ix[:,3], linestyle = "--", marker = "o", color = "g", label = "Centralized")
plt.plot(f_nu_nemps[h_nu_nemps.Cr == 1].NEmps, f_nu_nemps[h_nu_nemps.Cr == 1].ix[:,3], linestyle = "-", marker = "^", color = "g", label = "Decentralized")
ax2.legend(loc = "best", fontsize = 11)
plt.plot(h_nu_nemps[h_nu_nemps.Cr == 2].NEmps, h_nu_nemps[h_nu_nemps.Cr == 2].ix[:,3], linestyle = "--", marker = "o", color = "r", label = "Centralized")
plt.plot(f_nu_nemps[h_nu_nemps.Cr == 2].NEmps, f_nu_nemps[h_nu_nemps.Cr == 2].ix[:,3], linestyle = "-", marker = "^", color = "r", label = "Decentralized")
ax2.set_ylabel(r'$\nu$', fontsize = 20)
ax2.set_xlabel("N of Employees", fontsize = 16)

ax3 = fig.add_subplot(2, 2, 3)
plt.plot(h_eta_nsigs.index, h_eta_nsigs[2], linestyle = "--", marker = "o", color = "g", label = "Centralized")
plt.plot(f_eta_nsigs.index, f_eta_nsigs[2], linestyle = "-", marker = "^", color = "r", label = "Decentralized")
ax3.legend(loc = "best", fontsize = 11)
ax3.set_ylabel(r'$\eta$', fontsize = 20)
ax3.set_xlabel("N of Signals", fontsize = 16)



ax4 = fig.add_subplot(2, 2, 4)
plt.plot(h_eta_nsigs.index, h_eta_nsigs[1], linestyle = "--", marker = "o", color = "g", label = "Centralized", linewidth = 1)
plt.plot(h_eta_nsigs.index, h_eta_nsigs[2], linestyle = "--", marker = "o", color = (0, 0.5, 0), label = "Centralized", linewidth = 3)
plt.plot(h_eta_nsigs.index, h_eta_nsigs[3], linestyle = "--", marker = "o", color = "g", label = "Centralized", linewidth = 1)

plt.plot(f_eta_nsigs.index, f_eta_nsigs[1], linestyle = "-", marker = "^", color = "r", label = "Decentralized", linewidth = 1)
plt.plot(f_eta_nsigs.index, f_eta_nsigs[2], linestyle = "-", marker = "^", color = (0.5, 0, 0), label = "Decentralized", linewidth = 3)
plt.plot(f_eta_nsigs.index, f_eta_nsigs[3], linestyle = "-", marker = "^", color = "r", label = "Decentralized", linewidth = 1)
#ax4.legend(loc = "best", fontsize = 11)
ax4.set_ylabel(r'$\eta$', fontsize = 20)
ax4.set_xlabel("N of Signals", fontsize = 16)


#print(x_flat.describe())
#print(x_hier.describe())

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
