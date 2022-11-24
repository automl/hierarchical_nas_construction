from neps.search_spaces.graph_grammar.topologies import AbstractTopology

class BinaryTopo(AbstractTopology):
    edge_list = [(4, 5), (1, 2), (1, 3), (2, 4), (3, 4)]

    def __init__(self, *edge_vals):
        super().__init__()

        self.name = f"binary_op_{edge_vals[0]}"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)
        self.graph_type = "edge_attr"
