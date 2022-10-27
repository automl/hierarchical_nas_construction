from neps.search_spaces.graph_grammar.topologies import AbstractTopology


class NASBench201Cell(AbstractTopology):
    edge_list = [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4)]

    def __init__(self, *edge_vals):
        super().__init__()

        self.name = "cell"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))

        # Assign dummy variables as node attributes:
        for i in self.nodes:
            self.nodes[i]["op_name"] = "1"
        self.graph_type = "edge_attr"
        self.set_scope(self.name, recursively=False)


class Residual3(AbstractTopology):
    edge_list = [(1, 2), (2, 3), (1, 4), (3, 4)]

    def __init__(self, *edge_vals):
        super().__init__()

        self.name = "residual_3"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name, recursively=False)


class Diamond3(AbstractTopology):
    edge_list = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6)]

    def __init__(self, *edge_vals):
        super().__init__()
        self.name = "diamond_3"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name, recursively=False)
