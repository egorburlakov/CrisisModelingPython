import numpy as np
import networkx as nx
import pandas as pd
import bisect #for insort_left
import time
from datetime import datetime
import matplotlib.pyplot as plt
from CrisisModel import Crisis
from OrgModel import Org

cols = ["NInf", "NSigs", "NSigCaught", "NMissed", "NDecMade", "ImpCaught", "ImpMissed", "ImpDecMade", "NEmps", "MinSpan", "MaxSpan", "MaxLev", "CrT", "TpT"]
rec = {"NInf" : [], "NSigs" : [], "NSigCaught" : [], "NMissed" : [], "NDecMade" : [], "ImpCaught" : [], "ImpMissed" : [], "ImpDecMade" : [], "NEmps" : [],
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
       # print "{} {}".format("Start 1", time.time() - start)
        return Event(emp["t0"] + emp["t_proc"] + 1, e.sig_id, e.emp_id)

    s0 = cr.Sigs[e.sig_id]
    if np.random.choice(2, 1, p = [1 - cr.av, cr.av]): #otherwise he tries to catch the signal
        e1 = Event(e.t + emp["t_proc"] + 1, e.sig_id, e.emp_id, emp["stat"] + 1) #and moves to the next status if succeeds
        emp = nextStat(o, emp, e, emp["stat"] + 1)
        imp_caught = s0["imp"]
        if imp_caught >= cr.noise_th:
            rec["NSigCaught"][-1] += 1 #statistics
            rec["ImpCaught"][-1] += imp_caught
        return e1
    elif e.t < s0["app"] + s0["dapp"]:
        return Event(e.t + 1, e.sig_id, e.emp_id) #if he doesn't succeed to catch, he tries to catch the signal the next time step if the signal is still available
    else: #if signal disappears
        imp_caught = s0["imp"]
        if imp_caught >= cr.noise_th:
            rec["NMissed"][-1] += 1 #statistics
            rec["ImpMissed"][-1] += imp_caught
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
    emp_n = nx.shortest_path(o.g, e.emp_id, cr.Sigs[e.sig_id]["ag"])[1] #next angent in the path, might be similar to previous if he is a decision maker
    e1 = Event(e.t + emp["t_proc"] + 1, e.sig_id, emp_n, 1) #catch or evaluate?
    emp = nextStat(o, emp, e, 0) #the employee returns to monitoring status
    return e1

def actmonSig(e, cr, o, emp):
    s0 = cr.Sigs[e.sig_id]
    imp = s0["imp"] # the employees knows the true importance
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
    imp_caught = cr.Sigs[e.sig_id]["imp"]
    if imp_caught >= cr.noise_th:
        rec["NDecMade"][-1] += 1 #statistics
        rec["ImpDecMade"][-1] += imp_caught
    return []

handleEvent = {0 : catchSig, 1 : evalSig, 2 : transSig, 3 : actmonSig, 4 : prepSig, }

def runModeling(cr, o):
    e = [Event(s[1]["app"], s[0], s[1]["inf"]) for s in cr.Sigs.items()] #initializing events
    e.sort()
    e.append(Event(e[-1].t + cr.Sigs[e[-1].sig_id]["dapp"] + 1, -1, -1)) #adding terminal event
    while e:
        e1 = e.pop(0)
        if(e1.sig_id >= 0):
           # print "{} {} {}".format(e1.s_stat, "Start", time.time() - start)
            e1 = handleEvent[e1.s_stat](e1, cr, o, o.g.node[e1.emp_id])
            if e1:
         #       print "{} {} {}".format(e1.s_stat, "Event handled", time.time() - start)
                bisect.insort_right(e, e1)
                if gen_pars["VizOrg"] & (e1.s_stat > 0): o.visualizeGraph()
        else:
            e = []
    return e1.t

def runExperiments():
    for i in xrange(gen_pars["NExp"]):
        #generate org
        nemps = np.random.randint(org_pars[gen_pars["Scen"]]["NEmps_min"], org_pars[gen_pars["Scen"]]["NEmps_max"] + 1)
        rec["NEmps"].append(nemps)
        min_span = org_pars[gen_pars["Scen"]]["SpanMin"]
        max_span = org_pars[gen_pars["Scen"]]["SpanMax"]
        rec["MinSpan"].append(min_span)
        rec["MaxSpan"].append(max_span)
        o = Org(nemps, min_span, max_span)
        if gen_pars["VizOrg"]: o.visualizeGraph()
        rec["MaxLev"].append(o.max_lev)

        #generate crisis
        nsigs = np.random.randint(cr_pars["NSig_min"], cr_pars["NSig_max"] + 1)
        imp = np.random.choice(cr_pars["Imp_modes"], p = p_v, size = 1)
        rec["NSigs"].append(nsigs)
        for key in ["NSigCaught", "NMissed", "NDecMade", "ImpCaught", "ImpMissed", "ImpDecMade"]: #initialize stats
            rec[key].append(0)
        cr = Crisis(nsigs, cr_pars["AppT"], cr_pars["DappT"], imp, o) #nsigs, app, dapp, imp
        rec["NInf"].append(len(cr.Sigs))
        rec["CrT"].append(runModeling(cr, o)) # <-------- run modeling
        if i % 100 == 0: print "{} {} {} {} {} {}".format("Modeling #", i, "lasted for", rec["CrT"][-1], "and took", time.time() - start)
        for key in ["ImpCaught", "ImpMissed", "ImpDecMade"]: #normalize stats
            rec[key][-1] /= cr.imp_tot
        rec["TpT"] = o.g.node[0]["t_proc_tot"]

def vizVector(p, sb_plot, col):
    fig.add_subplot(sb_plot)
    plt.plot(p.index, p, linestyle = "--", marker = "o", color = col)
    plt.grid()

#########################`
#Run Experiment
#########################
#experiment details
gen_pars = {"Seed" : 2, "Scen" : "Nasa", "NExp" : 50, "ToCSV" : False, "VizOrg" : False}
org_pars = {"Nasa" : {"NEmps_min" : 22000, "NEmps_max" : 22500, "SpanMin" : 2, "SpanMax" : 8},
            "SmallOrg" : {"NEmps_min" : 5, "NEmps_max" : 19, "SpanMin" : 2, "SpanMax" : 3},
            "MiddleOrg" : {"NEmps_min" : 100, "NEmps_max" : 250, "SpanMin" : 2, "SpanMax" : 4},
            "BigOrg" : {"NEmps_min" : 250, "NEmps_max" : 1000, "SpanMin" : 2, "SpanMax" : 5}}
cr_pars = {"NSig_min" : 5, "NSig_max" : 11, "AppT" : 90, "DappT" : 30, "Imp_modes" : [0.45, 0.55]}
p_v = np.ones(len(cr_pars["Imp_modes"])) / len(cr_pars["Imp_modes"])

#Run experiments
np.random.seed(gen_pars["Seed"])
start = time.time()
runExperiments()
st = pd.DataFrame(rec, columns = cols)
#print st
print "{} {}".format("Execution time is", (time.time() - start))
if gen_pars["ToCSV"]: st.to_csv("{}_{}_{}.{}".format("ModelingResults", gen_pars["Scne"], datetime.now().strftime("%Y-%m-%d %H_%M_%S"), "csv"), sep=';')

#########################`
#Vizualize results
#########################`
fig = plt.figure(facecolor = "white")

#print st["ImpCaught"].groupby([st["NSigs"], rec["NEmps"]]).mean().unstack()
p1 = st["ImpCaught"].groupby([rec["NEmps"]]).mean()
vizVector(p1, 121, "g")
p2 = st["ImpCaught"].groupby([rec["NSigs"]]).mean()
vizVector(p2, 122, "r")
plt.show()