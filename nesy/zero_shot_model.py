import re
import numpy as np
import torch
import torch.nn as nn
from proScript.utils import GraphEditDistance
from feature_extraction import extract_text_features


class ZeroShotBase(nn.Module):
    def __init__(self, text_model):
        super(ZeroShotBase, self).__init__()
        # TODO: generalize to 'k' bounding boxes instead of 2
        self.text_model = text_model
        self.text_model.eval()
        self.sim_score = nn.CosineSimilarity(dim=1)
        self.w = nn.Sequential(nn.Linear(1,1, bias=True),
                               nn.Sigmoid())
        self.all_sorts = []

    def prompt(self, seg_text_feat, vid_feature):
        """
        given segment video features and query features
        output a logit for querying the segment with the text features
        """
        # computes similarity between seg_text_feat and vid_feature
        return self.sim_score(seg_text_feat, vid_feature)

    def process_nodes(self, nodes):
        """
        nodes: nodes of graph in NL
        """
        pred_args = []
        for node in nodes:
            node = re.sub('Step \d+ ', '', node).strip()
            pred_args.append(node)
        return pred_args

    @classmethod
    def all_topo_sorts(cls, graph):
        # get all possible topological sortings of the graphs
        nodes = graph.nodes
        edges = set(graph.edges)
        adj_mat = {k: [] for k in nodes}
        in_degree = {k: 0 for k in nodes}
        for edge in edges:
            src_node, dest_node, _ = edge
            adj_mat[src_node].append(dest_node)
            in_degree[dest_node] += 1

        visited = {k: False for k in nodes}
        curr_path = []
        cls.all_sorts = []
        cls.all_topo_sorts_util(nodes, edges, adj_mat, in_degree, visited, curr_path)
        return cls.all_sorts

    @classmethod
    def all_topo_sorts_util(cls, nodes, edges, adj_mat, in_degree, visited, curr_path):
        for node in nodes:
            if in_degree[node] == 0 and not visited[node]:
                for adj_n in adj_mat[node]:
                    in_degree[adj_n] -= 1

                curr_path.append(node)
                visited[node] = True
                cls.all_topo_sorts_util(nodes, edges, adj_mat,
                                        in_degree, visited, curr_path)

                # backtrack
                for adj_n in adj_mat[node]:
                    in_degree[adj_n] += 1
                curr_path.pop()
                visited[node] = False
        if len(curr_path) == len(nodes):
            cls.all_sorts.append(curr_path.copy())

    def dp_align(self, all_sorts, vid_feature):
        """
        optimized recurse: stores values of sub-problems in N*S array
        # nodes = N
        # segments = S
        """
        max_arr = [[]] * len(all_sorts)  # keeps track of max for each sorted sequence
        parent_dict = [[]] * len(all_sorts)  # keeps track of the best path for each sorted sequence
        num_segments = len(vid_feature[0])
        score_arr = [[]] * len(all_sorts)  # keeps track of similarity score matrix (for each cell)
        for ind, sorted_nodes in enumerate(all_sorts):
            nodes = all_sorts[ind]
            pred_args = self.process_nodes(nodes)
            with torch.no_grad():
                segment_text_feats = extract_text_features(pred_args, self.text_model, 'clip', tokenizer=None)
            seg_text_feats, seg_text_lens = segment_text_feats
            seg_text_feats = torch.stack([feat[:int(length.item())].sum(dim=0) for feat, length in zip(seg_text_feats,
                                                                                                       seg_text_lens)])
            # _, seg_text_feats = self.text_ctx_rnn(seg_text_feats, seg_text_lens)
            seg_text_feats = seg_text_feats.unsqueeze(0)  # [1, num_nodes, 512]

            num_nodes = len(sorted_nodes)
            parent_dict[ind] = {k1: {k2: tuple() for k2 in range(num_segments)} for k1 in sorted_nodes}
            # array keeps track of cumulative max logprob for each cell
            arr = torch.full((num_nodes, num_segments), torch.tensor(-100.)).cuda()
            score_arr[ind] = torch.zeros((num_nodes, num_segments)).cuda()

            # setting the start & end indices of each node on segments
            start_ind = dict(zip(sorted_nodes, np.arange(0, num_nodes, 1)))
            end_ind = dict(zip(sorted_nodes, [num_segments - num_nodes + i for i in range(num_nodes)]))

            # starting outer loop from the last node
            for node_ind, node in zip(np.arange(num_nodes - 1, -1, -1), reversed(sorted_nodes)):
                # starting inner loop from the last segment
                for segment_ind in range(end_ind[node], start_ind[node] - 1, -1):

                    # setting the value of the last column
                    if segment_ind == num_segments - 1:
                        score = self.prompt(seg_text_feats[:, node_ind, :],
                                            vid_feature[:, segment_ind, :])
                        arr[node_ind][segment_ind] = torch.log(score)
                        score_arr[ind][node_ind][segment_ind] = torch.log(score)
                        parent_dict[ind][node][segment_ind] = (segment_ind,)
                        continue

                    score = self.prompt(seg_text_feats[:, node_ind, :],
                                        vid_feature[:, segment_ind, :])

                    # setting the values of the last row
                    if node_ind == num_nodes - 1:
                        V_opt_curr = torch.log(score)
                        V_opt_next = arr[node_ind][segment_ind + 1]
                        if V_opt_curr >= V_opt_next:
                            arr[node_ind][segment_ind] = V_opt_curr
                            parent_dict[ind][node][segment_ind] = (segment_ind,)
                        else:
                            arr[node_ind][segment_ind] = V_opt_next
                            parent_dict[ind][node][segment_ind] = \
                                parent_dict[ind][sorted_nodes[node_ind]][segment_ind + 1]

                    # calculating the values of the remaining cells
                    # dp[i][j] = max(query(i,j) + dp[i+1][j], dp[i][j+1])
                    else:
                        V_opt_curr = torch.log(score) + arr[node_ind + 1][segment_ind]  # relaxation added
                        # V_opt_curr = score + arr[node_ind + 1][segment_ind + 1]  # no relaxation
                        V_opt_next = arr[node_ind][segment_ind + 1]
                        if V_opt_curr >= V_opt_next:
                            arr[node_ind][segment_ind] = V_opt_curr
                            parent_dict[ind][node][segment_ind] = \
                                (segment_ind,) + parent_dict[ind][sorted_nodes[node_ind + 1]][segment_ind + 1]
                        else:
                            arr[node_ind][segment_ind] = V_opt_next
                            parent_dict[ind][node][segment_ind] = \
                                parent_dict[ind][sorted_nodes[node_ind]][segment_ind + 1]
                    score_arr[ind][node_ind][segment_ind] = torch.log(score)

            max_arr[ind] = arr[0][0]
        max_sort_ind = torch.tensor(max_arr).argmax()
        # TODO: could be more than one optimum paths
        best_alignment = parent_dict[max_sort_ind][all_sorts[max_sort_ind][0]][0]
        aggregated_score = torch.tensor(0.).cuda()
        for i, j in zip(np.arange(num_nodes), best_alignment):
            aggregated_score += score_arr[max_sort_ind][i][j] / len(all_sorts[0])
        return max_sort_ind, max_arr[max_sort_ind], \
               list(zip(all_sorts[max_sort_ind], best_alignment)), aggregated_score

    def forward(self, vid_feats, graphs, true_labels, train=True):
        ent_probs = []
        labels = []
        pred_alignments = []
        for vid_feat, graph, hypothesis, label in zip(vid_feats, *graphs, true_labels):
            # processing the video features
            # each vid_feat is [num_segments, frames_per_segment, 512]
            b, vid_len, _ = vid_feat.shape
            # aggregate with positional embeddings?
            vid_feat = vid_feat.sum(dim=1)  # aggregate
            vid_feat = vid_feat.unsqueeze(0)  # [1, num_segments, 512]

            # dynamic programming
            try:
                all_sorts = ZeroShotBase.all_topo_sorts(graph)
                sorted_seq_ind, best_score, best_alignment, aligned_aggregated = self.dp_align(all_sorts, vid_feat)
                pred_alignments.append(best_alignment)
            except:
                print(hypothesis)
                print(GraphEditDistance.nx_to_string(graph))
                continue

            ent_probs.append(self.w(aligned_aggregated.unsqueeze(0)))
            labels.append(label)

        if not train:
            return torch.stack(ent_probs).view(-1), torch.stack(labels), pred_alignments
        return torch.stack(ent_probs).view(-1), torch.stack(labels)
