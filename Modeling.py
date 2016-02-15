import numpy as np
import networkx as nx
import pylab
import bisect #for insort_left
import CrisisModel.py
import OrgModel.py

class Event(object):
    def __init__(self, t_e, sig_e, stat_e, emp_e):
        self.t = t_e
        self.sig_id = sig_e
        self.stat = stat_e
        self.emp = emp_e

    def __eq__(self, other):
        return self.t == other.t

    def __lt__(self, other):
        return self.t < other.t

    def __repr__(self):
        return [self.t, self.sig_id].__repr__()


def catchSig(e, cr, o):
    if e.emp["stat"] != 0: #If the employee is busy we wait until he is free
        return Event(e.emp["t0"] + e.emp["t_proc"] + 1, e.sig_id, e.stat, e.emp)
    elif np.random.binomial(1, cr.av): #otherwise he tries to catch the signal
        print "Signal " + " caught!"
        e.emp["stat"] = e.emp["stat"] + 1
        e.emp["t0"] = e.t + 1
        e.emp["t_proc"] = o.t_proc[e.emp["stat"]]
        return Event(e.t + 1, e.sig_id, e.stat + 1, e.emp) #and moves to the next status if succeeds
    else: #if he doesn't succeed to catch, he tries to catch the signal the next time step
        print "Signal is missed!"
        return Event(e.t + 1, e.sig_id, e.stat, e.emp)

def evalSig(e, cr, o):
    print "evalSig"
    return []

def transSig(e, cr, o):
    print "transSig"
    return []

def monSig(e, cr, o):
    print "monSig"
    return []

def prepSig(e, cr, o):
    print "prepSig"
    return []

handleSig = {0 : catchSig, 1 : evalSig, 2 : transSig, 3 : monSig, 4 : prepSig}

def runModeling(cr, o):
    e = [Event(cr.Sigs[i]["app"], i, 0, o.g.node[cr.Sigs[i]["inf"]])
            for i in xrange(cr.Sigs.__len__())] #initializing events
    e.sort()
    while e:
        e1 = e.pop(0)
        e1 = handleSig[e1.stat](e1, cr, o)
        if e1: bisect.insort_right(e, e1)
    return []

#########################`
#Testing
#########################
o = Org(20, 2, 5)
cr = Crisis(10, 150, 30, 0.5, o)

#Visualize graph
nx.draw_spring(o.g, nodecolor = 'r', edge_color = 'b')
pylab.show()