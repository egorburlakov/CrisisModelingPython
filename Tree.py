import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pylab

class Agent(object):
    def __init__(self, id):
        self.id = id
        self.stat = np.random.randint(0, 6) #0 #0 - monitoring, 1 - catching, 2 - analysing, 3 - transferring, 4 - active monitoring, 5 - prepare
        self.t0 = 0

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'Agent #{} '.format(self.id) + '{}'.format(self.stat)



g = nx.balanced_tree(3, 3)
print g.nodes()

nx.relabel_nodes(g, mapping = Agent, copy = False)
print g.nodes()

#for n1 in g:
#    print n1.__repr__()
# print g.nodes()[1] == Agent(7)

nx.draw_spring(g, nodecolor = 'r', edge_color = 'b')
# pos = nx.graphviz_layout(g, prog = 'dot')
# nx.draw(g, pos, with_labels = False, arrows = False)
pylab.show()