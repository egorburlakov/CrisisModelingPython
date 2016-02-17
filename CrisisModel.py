import numpy as np

def getAllPred(o, n): #o - org, g - graph, n - node
    all_pred = []
    while o.g.predecessors(n):
        all_pred.append(o.g.predecessors(n)[0])
        n = all_pred[-1]
    return all_pred

class Crisis(object):
    av = 0.4 #probability that the signal will be caught
    imp_tot = 0

    def genSig(self, app, dapp, imp, o): #imp goes from 0.1 to 0.6
        s_app = round(np.random.exponential(app))
        s_dapp = max(round(np.random.exponential(dapp)), 1)
        s_imp = min(np.random.gamma(1, imp), 1)
        self.imp_tot += s_imp

        inf = np.random.randint(1, o.g.number_of_nodes()) #top cannot be a target for the signal
        ag = np.random.choice(getAllPred(o, inf)) #decision maker
        return {"app" : s_app, "dapp" : s_dapp, "imp" : s_imp, "imp_eval" : [], "inf" : inf, "ag" : ag, "av" : self.av} #if needed - can add emp_eval

    def __init__(self, nsigs, app, dapp, imp, o): # # of sigs, t of appearance, mean t for the lifespan of signals, mean importance, org
        self.Sigs = {} #dict of lists
        for i in xrange(nsigs):
            self.Sigs[i] = self.genSig(app, dapp, imp, o)  #id of the signal goes first
