import numpy as np
import networkx as nx

class Crisis(object):
    av = 0.05 #probability that the signal will be caught
    noise_th = 0.5 #threshold for signals to be not noise
    imp_tot = 0

    def genSig(self, app, dapp, imp, o): #imp goes from 0.1 to 0.6
        s_app = round(np.random.exponential(app))
        s_dapp = max(round(np.random.exponential(dapp)), 1)
        s_imp = min(np.random.gamma(1, imp), 1)
        if s_imp >= self.noise_th: self.imp_tot += s_imp

        inf = np.random.randint(1, o.g.number_of_nodes()) #top cannot be a target for the signal
        ag = np.random.choice(nx.shortest_path(o.g, inf, 0)) #decision maker
        return {"app" : s_app, "dapp" : s_dapp, "imp" : s_imp, "imp_eval" : [], "inf" : inf, "ag" : ag, "av" : self.av} #if needed - can add emp_eval

    def __init__(self, nsigs, app, dapp, imp, o): # # of sigs, t of appearance, mean t for the lifespan of signals, mean importance, org
        self.Sigs = {} #dict of lists
        i = j = 0
        while i < nsigs:
            self.Sigs[j] = self.genSig(app, dapp, imp, o)  #id of the signal goes first
            if self.Sigs[j]["imp"] >= self.noise_th: i += 1
            j += 1
