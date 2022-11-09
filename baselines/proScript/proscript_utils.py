import networkx as nx
import pydot
import torch
from torchmetrics import Metric


class GraphEditDistance(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.preds = []
        # self.targets = []
        self.add_state("preds", default=[])
        self.add_state("targets", default=[])
        self.add_state("dist", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pred, target):
        assert len(pred) == len(target)
        self.preds.extend(pred)
        self.targets.extend(target)
        self.total += len(target)

    def node_subst_cost(self, node1, node2):
        # check if the nodes are equal, if yes then apply no cost, else apply 1
        if node1 == node2:
            return torch.tensor(0.)
        return torch.tensor(1.)

    def node_del_cost(self):
        return torch.tensor(1.)  # here you apply the cost for node deletion

    def node_ins_cost(self):
        return torch.tensor(1.)  # here you apply the cost for node insertion

    # arguments for edges
    def edge_subst_cost(self, edge1, edge2):
        # check if the edges are equal, if yes then apply no cost, else apply 3
        if edge1 == edge2:
            return torch.tensor(0.)
        return torch.tensor(1.)

    def edge_del_cost(self):
        return torch.tensor(1.)  # here you apply the cost for edge deletion

    def edge_ins_cost(self):
        return torch.tensor(1.)  # here you apply the cost for edge insertion

    def pydot_to_nx(self, G_str):
        # string to pydot graph
        # G_str = G_str.replace("\n", "")
        G_str = "digraph graphDSL {" + G_str + "}"
        # G_str.replace(';', ';\n').replace('{', '{\n').replace('}', '}\n')
        G = pydot.graph_from_dot_data(G_str)[0]
        G.set_type('digraph')
        # pydot graph to networkx graph
        return nx.drawing.nx_pydot.from_pydot(G)

    @staticmethod
    def generate_adj_mat(nx_graph):
        # adjacency matrix from networkx graph
        return nx.to_numpy_array(nx_graph)

    def graph_edit_distance(self, G1, G2):
        G1 = self.pydot_to_nx(G1)
        G2 = self.pydot_to_nx(G2)
        return nx.graph_edit_distance(G1, G2,
                                      node_subst_cost=self.node_subst_cost,
                                      node_del_cost=self.node_del_cost,
                                      node_ins_cost=self.node_ins_cost,
                                      edge_subst_cost=self.edge_subst_cost,
                                      edge_del_cost=self.edge_del_cost,
                                      edge_ins_cost=self.edge_ins_cost)

    def compute(self, reinforce=False):
        # dist = []
        # if reinforce:
        # dist.append(self.graph_edit_distance(pred, target))
        if reinforce:
            return [-1. if x != y else 0. for (x, y) in zip(self.preds, self.targets)]
        else:
            for pred, target in zip(self.preds, self.targets):
                self.dist += self.graph_edit_distance(pred, target) / self.total
        return self.dist


def pred_args_map():
    return {'heat': 'hot', 'cool': 'cold', 'slice': 'sliced',
            'clean': 'clean', 'place': 'inReceptacle', 'pick': 'holds'}
