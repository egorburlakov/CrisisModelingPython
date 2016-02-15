def getAllPred(o, n): #o - org, g - graph, n - node
    all_pred = []
    while o.g.predecessors(n):
        all_pred.append(o.g.predecessors(n)[0])
        n = all_pred[-1]
    return all_pred

class Crisis(object):
    av = 0.05 #probability that the signal will be caught

    def genSig(self, app, dapp, imp, o): #imp goes from 0.1 to 0.6
        s_app = round(np.random.exponential(app))
        s_dapp = round(np.random.exponential(dapp))
        s_imp = np.random.gamma(1, imp)

        inf = np.random.randint(1, o.g.number_of_nodes()) #top cannot be a target for the signal
        ag = np.random.choice(getAllPred(o, inf)) #decision maker
        return {"app" : s_app, "dapp" : s_dapp, "imp" : s_imp, "inf" : inf, "ag" : ag, "av" : self.av}

    def __init__(self, nsigs, app, dapp, imp, o): # # of sigs, mean t between signals appearance, mean t for the lifespan of signals, mean importance, org
        self.Sigs = {} #dict of lists
        for i in xrange(nsigs):
            self.Sigs[i] = self.genSig(app, dapp, imp, o)  #id of the signal goes first
