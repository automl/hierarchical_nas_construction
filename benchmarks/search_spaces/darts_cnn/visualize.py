import os

from graphviz import Digraph
from path import Path


def plot(genotype, filename):
    g = Digraph(
        format="pdf",
        edge_attr=dict(fontsize="20", fontname="times"),
        node_attr=dict(
            style="filled",
            shape="rect",
            align="center",
            fontsize="20",
            height="0.5",
            width="0.5",
            penwidth="2",
            fontname="times",
        ),
        engine="dot",
    )
    g.body.extend(["rankdir=LR"])

    g.node("c_{k-2}", fillcolor="darkseagreen2")
    g.node("c_{k-1}", fillcolor="darkseagreen2")
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor="lightblue")

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor="palegoldenrod")
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename, view=True)
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    os.remove(dir_path / filename)


def plot_from_graph(graph, filename, node_attr: bool = True):
    g = Digraph(
        format="pdf",
        edge_attr=dict(fontsize="20", fontname="times"),
        node_attr=dict(
            style="filled",
            shape="rect",
            align="center",
            fontsize="20",
            height="0.5",
            width="0.5",
            penwidth="2",
            fontname="times",
        ),
        engine="dot",
    )
    g.body.extend(["rankdir=LR"])

    if node_attr:
        for n, data in graph.nodes(data=True):
            op_name = data["op_name"]
            if "input" in op_name:
                g.node(str(n), op_name, fillcolor="darkseagreen2")
            elif "output" == op_name:
                g.node(str(n), op_name, fillcolor="palegoldenrod")
            elif op_name == "add":
                g.node(str(n), op_name, fillcolor="gray", shape="ellipse")
            else:
                g.node(str(n), op_name, fillcolor="lightblue", shape="ellipse")
        for u, v in graph.edges:
            g.edge(str(u), str(v), fillcolor="gray")
    else:
        raise NotImplementedError

    g.render(filename, view=True)
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    os.remove(dir_path / filename)
