import numpy as np
import networkx as nx
import pandas as pd
import bisect #for insort_left
import time
import matplotlib.pyplot as plt
from CrisisModel import Crisis
from OrgModel import Org

viz = False
cols = ["NSigs", "NSigCaught", "NMissed", "NDecMade", "ImpCaught", "ImpMissed", "ImpDecMade", "NEmps", "MinSpan", "MaxSpan", "MaxLev", "CrT", "TpT"]
rec = {"NSigs" : [], "NSigCaught" : [], "NMissed" : [], "NDecMade" : [], "ImpCaught" : [], "ImpMissed" : [], "ImpDecMade" : [], "NEmps" : [],
       "MinSpan" : [], "MaxSpan" : [], "MaxLev" : [], "CrT" : [], "TpT" : []} #record on one experiment

class Event(object):
    def __init__(self, t_e, sig_id_e, emp_id_e, s_st = 0):  #stat - status of the signal to be processed
        self.t = t_e #time when the event happens
        self.sig_id, self.emp_id, self.s_stat = sig_id_e, emp_id_e, s_st

    def __eq__(self, other):
        return self.t == other.t
    def __lt__(self, other):
        return self.t < other.t
    def __repr__(self):
        return [self.t, self.sig_id, self.emp_id, self.s_stat].__repr__()

def nextStat(o, emp, e, stat_n): #sets employees pars ready for the next stat
    emp["t0"] = e.t + emp["t_proc"] + 1
    emp["stat"] = stat_n
    emp["t_proc"] = o.t_proc[stat_n]
    emp["sig_proc"] = e.sig_id if stat_n > 0 else -1
    emp["t_proc_tot"] += emp["t_proc"]
    return emp

def catchSig(e, cr, o, emp):
    if emp["stat"] != 0: #If the employee is busy we wait until he is free
        return Event(emp["t0"] + emp["t_proc"] + 1, e.sig_id, e.emp_id)
    elif np.random.choice(2, 1, p = [1 - cr.av, cr.av]): #otherwise he tries to catch the signal
        #print "{} {}".format("Signal", e.sig_id) + " is caught!"
        e1 = Event(e.t + emp["t_proc"] + 1, e.sig_id, e.emp_id, emp["stat"] + 1) #and moves to the next status if succeeds
        emp = nextStat(o, emp, e, emp["stat"] + 1)
        rec["NSigCaught"][-1] += 1 #statistics
        rec["ImpCaught"][-1] += cr.Sigs[e.sig_id]["imp"]
        return e1
    elif e.t < cr.Sigs[e.sig_id]["app"] + cr.Sigs[e.sig_id]["dapp"]:
        return Event(e.t + 1, e.sig_id, e.emp_id) #if he doesn't succeed to catch, he tries to catch the signal the next time step if the signal is still available
    #print "{} {}".format("Signal", e.sig_id) + " disappeared!"
    rec["NMissed"][-1] += 1 #statistics
    rec["ImpMissed"][-1] += cr.Sigs[e.sig_id]["imp"]
    return [] #otherwise the signal disappears

def evalSig(e, cr, o, emp):
    if (emp["sig_proc"] >= 0) & (emp["sig_proc"] != e.sig_id): #If the employee is busy we wait until he is free
        return Event(emp["t0"] + emp["t_proc"] + 1, e.sig_id, e.emp_id, 1)
    s0 = cr.Sigs[e.sig_id]
    if s0["imp_eval"]: #if someone has already assessed the signal
        imp = min(max(np.random.normal(s0["imp_eval"][-1], o.var_imp_eval), 0), 1)
    else:
        imp = min(max(np.random.normal(s0["imp"], o.var_imp_eval), 0), 1)
    s0["imp_eval"].append(imp) #update the list of evaluations from employees
    if imp > o.s2: #transfer signal
        if s0["ag"] == e.emp_id: #if it's the decision maker to preparation, otherwise - to transferring
            e1 = Event(e.t + emp["t_proc"] + 1, e.sig_id, e.emp_id, 4)
            emp = nextStat(o, emp, e, 4)
        else:
            e1 = Event(e.t + emp["t_proc"] + 1, e.sig_id, e.emp_id, emp["stat"] + 1)
            emp = nextStat(o, emp, e, emp["stat"] + 1)
        return e1
    elif imp >= o.s1: #active monitoring
        e1 = Event(e.t + emp["t_proc"] + 1, e.sig_id, e.emp_id, 3)
        emp = nextStat(o, emp, e, 3)
        return e1
    else: #don't process anymore
        emp = nextStat(o, emp, e, 0)
        return []

def transSig(e, cr, o, emp):
    emp_n = nx.shortest_path(o.g.to_undirected(), e.emp_id, cr.Sigs[e.sig_id]["ag"])[1] #next angent in the path, might be similar to previous if he is a decision maker
    e1 = Event(e.t + emp["t_proc"] + 1, e.sig_id, emp_n, 1) #catch or evaluate?
    emp = nextStat(o, emp, e, 0) #the employee returns to monitoring status
    return e1

def actmonSig(e, cr, o, emp):
    s0 = cr.Sigs[e.sig_id]
    imp = min(max(np.random.normal(s0["imp"], o.var_imp_eval), 0), 1)
    s0["imp_eval"].append(imp) #update the list of evaluations from employees
    if imp >= (o.s2 + o.s1) / 2: #transfer signal
        if s0["ag"] == e.emp_id: #if it's the target agent - preparation, otherwise - transferring
            e1 = Event(e.t + emp["t_proc"] + 1, e.sig_id, e.emp_id, 4)
            emp = nextStat(o, emp, e, 4)
        else:
            e1 = Event(e.t + emp["t_proc"] + 1, e.sig_id, e.emp_id, 2)
            emp = nextStat(o, emp, e, 2)
        return e1
    else: #don't process anymore
        emp = nextStat(o, emp, e, 0)
        return []

def prepSig(e, cr, o, emp):
    #print "{} {}".format("Signal", e.sig_id) + " is processed!"
    emp = nextStat(o, emp, e, 0)
    rec["NDecMade"][-1] += 1 #statistics
    rec["ImpDecMade"][-1] += cr.Sigs[e.sig_id]["imp"]
    return []

handleEvent = {0 : catchSig, 1 : evalSig, 2 : transSig, 3 : actmonSig, 4 : prepSig, }

def runModeling(cr, o):
    e = [Event(s[1]["app"], s[0], s[1]["inf"]) for s in cr.Sigs.items()] #initializing events
    e.sort()
    e.append(Event(e[-1].t + cr.Sigs[e[-1].sig_id]["dapp"] + 1, -1, -1)) #adding terminal event
    while e:
        e1 = e.pop(0)
        if(e1.sig_id >= 0):
            e1 = handleEvent[e1.s_stat](e1, cr, o, o.g.node[e1.emp_id])
            if e1:
                bisect.insort_right(e, e1)
                if viz & (e1.s_stat > 0): o.visualizeGraph()
        else:
            e = []
    return e1.t

def runExperiments():
    for i in xrange(n_exp):
        #generate org
        nemps = np.random.randint(nemps_min, nemps_max + 1)
        rec["NEmps"].append(nemps)
        rec["MinSpan"].append(min_span)
        rec["MaxSpan"].append(max_span)
        o = Org(nemps, min_span, max_span)
        #o.visualizeGraph()
        rec["MaxLev"].append(o.max_lev)

        #generate crisis
        nsigs = np.random.randint(nsigs_min, nsigs_max)
        imp = np.random.choice(imp_v, p = p_v, size = 1)
        rec["NSigs"].append(nsigs)
        for key in ["NSigCaught", "NMissed", "NDecMade", "ImpCaught", "ImpMissed", "ImpDecMade"]: #initialize stats
            rec[key].append(0)
        cr = Crisis(nsigs, 100, 30, imp, o) #nsigs, app, dapp, imp
        rec["CrT"].append(runModeling(cr, o)) # <-------- run modeling
        if i % 100 == 0: print "{} {} {} {}".format("Modeling #", i, "lasted for", rec["CrT"][-1])
        for key in ["ImpCaught", "ImpMissed", "ImpDecMade"]: #normalize stats
            rec[key][-1] /= cr.imp_tot
        rec["TpT"] = o.g.node[0]["t_proc_tot"]

#########################`
#Run Experiment
#########################
np.random.seed(2)

#experiment details
n_exp = 10000
nemps_min = 5
nemps_max = 500
min_span = 2
max_span = 3

nsigs_min = 3
nsigs_max = 12
imp_v = [0.4, 0.6, 0.8]
p_v = np.ones(imp_v.__len__()) / imp_v.__len__()

start = time.time()
runExperiments()
st = pd.DataFrame(rec, columns = cols)
end = time.time()
print "{} {}".format("Execution time is", (end - start))

######
#Viz
########

#print st["ImpCaught"].groupby([st["NSigs"], rec["NEmps"]]).mean().unstack()

p = st["ImpCaught"].groupby([rec["NEmps"]]).mean()
print p
fig = plt.figure(facecolor = "white")
ax2 = fig.add_subplot(1, 1, 1)
plt.plot(p.index, p, linestyle = "--", marker = "o", color = "g")
plt.grid()
plt.show()
