import networkx as nx
from neps.search_spaces.graph_grammar.topologies import AbstractTopology

from benchmarks.search_spaces.darts_cnn.primitives import (
    Concat,
    SkipConnect,
    Stacking,
    Unbinder,
)


class DARTSCell(AbstractTopology):
    def __init__(self, *edge_vals):
        super().__init__()
        self.name = "darts_cell"

        in_nodes = [int(val) for idx, val in enumerate(edge_vals) if idx % 2 == 1]
        self.edge_list = [
            (in_nodes[0], 2),
            (in_nodes[1], 2),
            (in_nodes[2], 3),
            (in_nodes[3], 3),
            (in_nodes[4], 4),
            (in_nodes[5], 4),
            (in_nodes[6], 5),
            (in_nodes[7], 5),
            (2, 6),
            (3, 6),
            (4, 6),
            (5, 6),
        ]
        edge_list = [(-1, 0), (-1, 1)] + self.edge_list  # -1 is an helper input node

        self.edge_vals = list(edge_vals)
        op_on_edge = [{"op": Unbinder, "idx": 0}, {"op": Unbinder, "idx": 1}]
        op_on_edge += [val for idx, val in enumerate(edge_vals) if idx % 2 == 0]
        op_on_edge += [{"op": SkipConnect, "C": None, "stride": 1} for _ in range(4)]

        self.create_graph(dict(zip(edge_list, op_on_edge)))

        # Assign dummy variables as node attributes:
        for i in self.nodes:
            self.nodes[i]["op_name"] = "1"

        self.nodes[-1].update({"comb_op": Stacking()})
        for idx in range(6):
            self.nodes[idx].update({"comb_op": sum})
        self.nodes[6].update({"comb_op": Concat()})

        self.graph_type = "edge_attr"
        self.set_scope(self.name, recursively=False)

    def get_node_list_and_ops(self):
        ops = [val["op_name"] for idx, val in enumerate(self.edge_vals) if idx % 2 == 0]
        in_nodes = [val for idx, val in enumerate(self.edge_vals) if idx % 2 == 1]
        cell = list(zip(ops, in_nodes))

        G = nx.DiGraph()
        n_nodes = (8 // 2) * 3 + 3
        G.add_nodes_from(range(n_nodes), op_name=None)
        n_ops = 8 // 2
        G.nodes[0]["op_name"] = "input1"
        G.nodes[1]["op_name"] = "input2"
        G.nodes[n_nodes - 1]["op_name"] = "output"
        for i in range(n_ops):
            G.nodes[i * 3 + 2]["op_name"] = cell[i * 2][0]
            G.nodes[i * 3 + 3]["op_name"] = cell[i * 2 + 1][0]
            G.nodes[i * 3 + 4]["op_name"] = "add"
            G.add_edge(i * 3 + 2, i * 3 + 4)
            G.add_edge(i * 3 + 3, i * 3 + 4)

        for i in range(n_ops):
            # Add the connections to the input
            for offset in range(2):
                if cell[i * 2 + offset][1] == 0:
                    G.add_edge(0, i * 3 + 2 + offset)
                elif cell[i * 2 + offset][1] == 1:
                    G.add_edge(1, i * 3 + 2 + offset)
                else:
                    k = cell[i * 2 + offset][1] - 2
                    # Add a connection from the output of another block
                    G.add_edge(int(k) * 3 + 4, i * 3 + 2 + offset)
        # Add connections to the output
        for i in range(2, 6):
            if i <= 1:
                G.add_edge(i, n_nodes - 1)  # Directly from either input to the output
            else:
                op_number = i - 2
                G.add_edge(op_number * 3 + 4, n_nodes - 1)
        # Remove the skip link nodes, do another sweep of the graph
        for j in range(n_nodes):
            try:
                G.nodes[j]
            except KeyError:
                continue
            if G.nodes[j]["op_name"] == "skip_connect":
                in_edges = list(G.in_edges(j))
                out_edge = list(G.out_edges(j))[0][
                    1
                ]  # There should be only one out edge really...
                for in_edge in in_edges:
                    G.add_edge(in_edge[0], out_edge)
                G.remove_node(j)
            elif G.nodes[j]["op_name"] == "none":
                G.remove_node(j)
        for j in range(n_nodes):
            try:
                G.nodes[j]
            except KeyError:
                continue

            if G.nodes[j]["op_name"] not in ["input1", "input2"]:
                # excepting the input nodes, if the node has no incoming edge, remove it
                if len(list(G.in_edges(j))) == 0:
                    G.remove_node(j)
            elif G.nodes[j]["op_name"] != "output":
                # excepting the output nodes, if the node has no outgoing edge, remove it
                if len(list(G.out_edges(j))) == 0:
                    G.remove_node(j)
            elif (
                G.nodes[j]["op_name"] == "add"
            ):  # If add has one incoming edge only, remove the node
                in_edges = list(G.in_edges(j))
                out_edges = list(G.out_edges(j))
                if len(in_edges) == 1 and len(out_edges) == 1:
                    G.add_edge(in_edges[0][0], out_edges[0][1])
                    G.remove_node(j)

        return G
