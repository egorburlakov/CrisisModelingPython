class Org(object):
    t_proc = {1 : 1, 2 : 50, 3 : 5, 4 : 100, 5 : 200}  #1 - catching, 2 - analysing, 3 - transferring, 4 - active monitoring, 5 - prepare
    var_imp_eval = 0.2 #variation in evaluation of the signal's importance
    p_err_transf = 0.1 #probability of error in the choice of the direction for transferring the signal

    def setNodePars(self, id, lev):
        self.g.node[id]["lev"] = lev
        self.g.node[id]["stat"] = 0 #0 - monitoring, 1 - catching, 2 - analysing, 3 - transferring, 4 - active monitoring, 5 - prepare
        self.g.node[id]["t0"] = 0 #time when he entered the status
        self.g.node[id]["t_proc"] = self.t_proc

    def addNodes(self, nodes_act, n_ch, n_ag_cur):
        if len(nodes_act) > 0: n_act = nodes_act.pop(0)

        for i in xrange(n_ch):
            self.g.add_edge(n_act, n_ag_cur + i)
            self.setNodePars(n_ag_cur + i, self.g.node[n_act]["lev"] + 1) #initializing parameters of the node
            nodes_act.append(n_ag_cur + i)
        return nodes_act

    def createGraph(self, n_ag, min_span, max_span):
        self.g = nx.DiGraph() #init graph
        self.g.add_node(0, lev = 0) #root
        self.setNodePars(0, 0)

        nodes_act = [self.g.nodes()[0]]
        n_ag_cur = self.g.number_of_nodes()
        while n_ag_cur < n_ag - max_span:
            n_ch = np.random.randint(min_span, max_span + 1) # how many nodes to add
            nodes_act = self.addNodes(nodes_act, n_ch, n_ag_cur)
            n_ag_cur = self.g.number_of_nodes()
        self.addNodes(nodes_act, n_ag - n_ag_cur, n_ag_cur) # last leafs to have the precise  number of agents

    def __init__(self, n_ag, min_span, max_span, sigma1 = 0.4, sigma2 = 0.6): #number of agents, min span of control, max span of control
        self.createGraph(n_ag, min_span, max_span)
        self.s1 = sigma1
        self.s2 = sigma2

