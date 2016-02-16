import numpy as np
import networkx as nx
import bisect #for insort_left
from CrisisModel import Crisis
from OrgModel import Org

class Event(object):
    def __init__(self, t_e, sig_id_e, emp_id_e):
        self.t = t_e #time when the event happens
        self.sig_id, self.emp_id = sig_id_e, emp_id_e

    def __eq__(self, other):
        return self.t == other.t
    def __lt__(self, other):
        return self.t < other.t
    def __repr__(self):
        return [self.t, self.sig_id, self.emp_id].__repr__()

def nextStat(o, emp, stat_n, t0_n): #sets employees pars ready for the next stat
    emp["stat"], emp["t0"]  = stat_n, t0_n
    emp["t_proc"] = o.t_proc[stat_n]
    return emp

def catchSig(e, cr, o, emp):
    #print [e.sig_id, e.t, cr.Sigs[e.sig_id]["app"], cr.Sigs[e.sig_id]["dapp"]]
    if emp["stat"] != 0: #If the employee is busy we wait until he is free
        return Event(emp["t0"] + emp["t_proc"] + 1, e.sig_id, e.emp_id)
    elif np.random.choice(2, 1, p = [1 - cr.av, cr.av]): #otherwise he tries to catch the signal
        print "{} {}".format("Signal", e.sig_id) + " is caught!"
        o.g.node[e.emp_id] = nextStat(o, emp, emp["stat"] + 1, e.t + 1)
        return Event(e.t + 1, e.sig_id, e.emp_id) #and moves to the next status if succeeds
    elif e.t < cr.Sigs[e.sig_id]["app"] + cr.Sigs[e.sig_id]["dapp"]:
        return Event(e.t + 1, e.sig_id, e.emp_id) #if he doesn't succeed to catch, he tries to catch the signal the next time step if the signal is still available
    print "{} {}".format("Signal", e.sig_id) + " disappeared!"
    return [] #otherwise the signal disappears

def evalSig(e, cr, o, emp):
    s0 = cr.Sigs[e.sig_id]
    if s0["imp_eval"]: #if someone has already assessed the signal
        imp = min(max(np.random.normal(s0["imp_eval"][-1], o.var_imp_eval), 0), 1)
    else:
        imp = min(max(np.random.normal(s0["imp"], o.var_imp_eval), 0), 1)
    cr.Sigs[e.sig_id]["imp_eval"].append(imp) #update the list of evaluations from employees
    if imp > o.s2: #transfer signal
        if s0["ag"] == e.emp_id:
            o.g.node[e.emp_id] = nextStat(o, emp, 4, e.t + 1) #to decision making
        else:
            o.g.node[e.emp_id] = nextStat(o, emp, emp["stat"] + 1, e.t + 1) #to transferring
        return Event(e.t + 1, e.sig_id, e.emp_id)
    elif imp >= o.s1: #active monitoring
        o.g.node[e.emp_id] = nextStat(o, emp, 3, e.t + 1)
        return Event(e.t + 1, e.sig_id, e.emp_id)
    else: #don't process anymore
        o.g.node[e.emp_id] = nextStat(o, emp, 0, e.t + 1)
        return []

def transSig(e, cr, o, emp):
    print "{} {}".format("Signal", e.sig_id) + " is transferred!"
    o.g.node[e.emp_id] = nextStat(o, emp, 0, e.t + 1)
    emp_n = nx.shortest_path(o.g.to_undirected(), e.emp_id, cr.Sigs[e.sig_id]["ag"])[1] #next angent in the path, might be similar to previous if he is a decision maker
    o.g.node[emp_n] = nextStat(o, o.g.node[emp_n], 0, e.t + 1) #to monitoring
    return Event(e.t + 1, e.sig_id, emp_n)

def actmonSig(e, cr, o, emp):
    s0 = cr.Sigs[e.sig_id]
    imp = min(max(np.random.normal(s0["imp"], o.var_imp_eval), 0), 1)
    cr.Sigs[e.sig_id]["imp_eval"].append(imp) #update the list of evaluations from employees
    if imp >= (o.s2 + o.s1) / 2: #transfer signal
        if s0["ag"] == e.emp_id:
            o.g.node[e.emp_id] = nextStat(o, emp, 4, e.t + 1) #to decision making
        else:
            o.g.node[e.emp_id] = nextStat(o, emp, 2, e.t + 1) #to transferring
        return Event(e.t + 1, e.sig_id, e.emp_id)
    else: #don't process anymore
        o.g.node[e.emp_id] = nextStat(o, emp, 0, e.t + 1)
        return []

def prepSig(e, cr, o, emp):
    print "{} {}".format("Signal", e.sig_id) + " is processed!"
    o.g.node[e.emp_id] = nextStat(o, emp, 0, e.t + 1)
    return []

handleEvent = {0 : catchSig, 1 : evalSig, 2 : transSig, 3 : actmonSig, 4 : prepSig, }

def runModeling(cr, o):
    e = [Event(s[1]["app"], s[0], s[1]["inf"]) for s in cr.Sigs.items()] #initializing events
    e.sort()
    e.append(Event(e[-1].t + cr.Sigs[e[-1].sig_id]["dapp"] + 1, -1, -1)) #adding terminal event
    while e:
        e1 = e.pop(0)
        if(e1.sig_id >= 0):
            e1 = handleEvent[o.g.node[e1.emp_id]['stat']](e1, cr, o, o.g.node[e1.emp_id])
            if e1:
                bisect.insort_right(e, e1)
                if o.g.node[e1.emp_id]["stat"] > 0: o.visualizeGraph()
        else:
            print "Modeling is finished!"
            e = []
    return []

#########################`
#Testing
#########################
o = Org(20, 2, 5)
cr = Crisis(10, 150, 3, 0.5, o) #nsigs, app, dapp, imp
runModeling(cr, o)
o.visualizeGraph()