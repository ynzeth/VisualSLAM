from ortools.graph import pywrapgraph
import numpy as np

from gui import showMatch

# class parameter similarities : |Q| x |D| matrix (np array) with similarity scores.
# class parameter K : hyperparameter that decides the 'spread' of connections between layers
# class parameter W : hyperparameter that is the constant weight of edges going into hidden nodes
# class parameter C : hyperparameter that is the cost a matching of similarity 1 will have

class matching:
    def __init__(self, similarities, K, W, C):
        self.similarities = similarities

        self.K = K
        self.W = W
        self.C = C

        self.solver = pywrapgraph.SimpleMinCostFlow()

        self.start_nodes = []
        self.end_nodes = []
        self.unit_costs = []
        self.capacities = []
        self.supplies = []

        self.initializeEdges()
        self.status = self.match()
        self.matchings = self.presentResult()

    def match(self):
        # add each arc
        for arc in zip(self.start_nodes, self.end_nodes, self.capacities, self.unit_costs):
            self.solver.AddArcWithCapacityAndUnitCost(arc[0],arc[1],arc[2],arc[3])

        # add node supply
        for node, supply in enumerate(self.supplies):
            self.solver.SetNodeSupply(node, supply)
        
        return self.solver.Solve()

    def getNodeId(self, Qid = 0, Did = 0, hidden = False, source = False, sink = False):
        if source:
            return 2 * (self.similarities.shape[0] * self.similarities.shape[1])
        if sink:
            return 2 * (self.similarities.shape[0] * self.similarities.shape[1]) + 1
        if hidden:
            return (self.similarities.shape[0] * self.similarities.shape[1]) + (Qid * self.similarities.shape[1]) + Did
        
        return (Qid * self.similarities.shape[1]) + Did

    # returns {queryId, databaseId} if the node is a matching node
    def getQueryDatabaseId(self, nodeId):
        if nodeId < self.similarities.shape[0] * self.similarities.shape[1]:
            databaseId = nodeId % self.similarities.shape[1]
            queryId = (nodeId - databaseId) / self.similarities.shape[1]
            return int(queryId), int(databaseId)
        return False

    def initializeEdges(self):
        for i in range(2 * (self.similarities.shape[0] * self.similarities.shape[1])):
            self.supplies.append(0)
                # supplies
        self.supplies.append(1)
        self.supplies.append(-1)

        # epsilon s (source to all)
        for i in range(2): # matching / hidden
            for j in range(self.similarities.shape[1]): # database image
                self.start_nodes.append(self.getNodeId(source=True))
                self.end_nodes.append(self.getNodeId(0, j, i))
                self.unit_costs.append(0)
                self.capacities.append(1)
        
        # epsilon t (all to sink)
        for i in range(2): # matching / hidden
            for j in range(self.similarities.shape[1]): # database image
                self.start_nodes.append(self.getNodeId(self.similarities.shape[0] - 1, j, i))
                self.end_nodes.append(self.getNodeId(sink=True))
                self.unit_costs.append(0)
                self.capacities.append(1)

        # epsilon a (matching to matching and hidden to hidden)
        for i in range(2): # matching / hidden
            for j in range(self.similarities.shape[0] - 1): # query image
                for k in range(self.similarities.shape[1] - 1): # database image
                    for l in range(self.K): # spread
                        if k+l < self.similarities.shape[1]:
                            self.start_nodes.append(self.getNodeId(j, k, i))
                            self.end_nodes.append(self.getNodeId(j+1, k+l, i))
                            if(i):
                                self.unit_costs.append(self.W)
                            else:
                                self.unit_costs.append(int(self.C / self.similarities[j+1][k+l]))
                            self.capacities.append(1)
        
        # epsilon b (matching to hidden and hidden to matching)
        for i in range(2): # matching / hidden
            for j in range(self.similarities.shape[0] - 1): # query image
                for k in range(self.similarities.shape[1] - 1): # database image
                    for l in range(self.K): # spread
                        if k+l < self.similarities.shape[1]:
                            if(i): # matching to hidden
                                self.start_nodes.append(self.getNodeId(j, k, not i))
                                self.end_nodes.append(self.getNodeId(j+1, k+l, i))
                                self.unit_costs.append(self.W)
                            else: # hidden to matching
                                self.start_nodes.append(self.getNodeId(j, k, i))
                                self.end_nodes.append(self.getNodeId(j+1, k+l, not i))
                                self.unit_costs.append(int(self.C / self.similarities[j+1][k+l]))
                            self.capacities.append(1)

        # epsilon h (horizontal connection)
        for i in range(2): # matching / hidden
            for j in range(self.similarities.shape[0]-1): # query image
                for k in range(self.similarities.shape[1] - 1): # database image
                    self.start_nodes.append(self.getNodeId(j, k, i))
                    self.end_nodes.append(self.getNodeId(j, k+1, i))
                    self.unit_costs.append(0)
                    self.capacities.append(1)
        
    def presentResult(self):
        if self.status != self.solver.OPTIMAL:
            print('There was an issue with the min cost flow input, status: ' + self.status)
            exit(1)

        matches = []

        print('Arc      Flow / Capacity   Cost')
        for i in range(self.solver.NumArcs()):
            cost = self.solver.Flow(i) * self.solver.UnitCost(i)
            if self.solver.Flow(i) > 0 and self.solver.UnitCost(i) > 0:
                if self.getQueryDatabaseId(self.solver.Head(i)) != False:
                    queryId, databaseId = self.getQueryDatabaseId(self.solver.Head(i))
                    matches.append([databaseId, queryId])
                    showMatch(queryId, databaseId)

                print( '%1s -> %1s    %3s   / %3s   %3s' % (self.solver.Tail(i), self.solver.Head(i), self.solver.Flow(i), self.solver.Capacity(i), cost) )
        
        print('Minimum cost: ', self.solver.OptimalCost())

        return matches


with open('similarityMatrix.npy', 'rb') as f:
    similarities = np.load(f)

matching = matching(similarities, 8, 130, 100)

with open('matchings.npy', 'wb') as f:
    np.save(f, matching.matchings)