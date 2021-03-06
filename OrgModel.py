import numpy as np
import pylab
import networkx as nx
import time

graph_colors = {0 : "grey", 1 : "green", 2 : "yellow", 3 : "orange", 4 : "red"}

start = time.time()

class Org(object):
    t_proc = {0 : 2, 1 : 16, 2 : 4, 3 : 40, 4 : 80}  #0 - catching, 1 - evaluating, 2 - transferring, 3 - active monitoring, 4 - prepare
    var_imp_eval = 0.2 #variation in evaluation of the signal's importance
    #p_err_transf = 0.1 #probability of error in the choice of the direction for transferring the signal
    max_lev = 0

    def setNodePars(self, id, lev):
        node = self.g.node[id]
        node["lev"] = lev
        node["stat"] = 0 #0 - monitoring, 1 - catching, 2 - analysing, 3 - transferring, 4 - active monitoring, 5 - prepare
        node["t0"] = 0 #time when he entered the status
        node["t_proc"] = self.t_proc[0]
        node["sig_proc"] = -1 #signal currently processing
        node["t_proc_tot"] = 0 #total time spent by the employee on signal processing

    def addNodes(self, nodes_act, n_ch, n_ag_cur):
        if len(nodes_act) > 0: n_act = nodes_act.pop(0)
        l_n = self.g.node[n_act]["lev"] + 1
        for i in xrange(n_ch):
            self.g.add_edge(n_act, n_ag_cur + i)
            self.setNodePars(n_ag_cur + i, l_n) #initializing parameters of the node
            nodes_act.append(n_ag_cur + i)
        if l_n > self.max_lev: self.max_lev = l_n
        return nodes_act

    def createGraph(self, n_ag, min_span, max_span):
        self.g = nx.Graph() #nx.DiGraph() <- very slowly #init graph
        self.g.add_node(0, lev = 0) #root
        self.setNodePars(0, 0)

        nodes_act = [self.g.nodes()[0]]
        n_ag_cur = self.g.number_of_nodes() #current number of nodes in the graph
        while n_ag_cur < n_ag - max_span:
            n_ch = np.random.randint(min_span, max_span + 1) # how many nodes to add
            nodes_act = self.addNodes(nodes_act, n_ch, n_ag_cur)
            n_ag_cur = self.g.number_of_nodes()
        self.addNodes(nodes_act, n_ag - n_ag_cur, n_ag_cur) # last leafs to have the precise  number of agents
        self.pos = nx.spring_layout(self.g) if self.viz else 0

    def __init__(self, n_ag, min_span, max_span, sigma1 = 0.4, sigma2 = 0.6, viz = False): #number of agents, min span of control, max span of control
        self.viz = viz
        self.createGraph(n_ag, min_span, max_span)
        self.s1, self.s2 = sigma1, sigma2

    def visualizeGraph(self):
        if self.viz:
            nodes_colors = [graph_colors[s["stat"]] for s in self.g.node.values()]
            nx.draw(self.g, pos = self.pos, node_color = nodes_colors, edge_color = 'b', with_labels = True)
            pylab.show()