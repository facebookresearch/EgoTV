import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from proScript.utils import GraphEditDistance
from feature_extraction import extract_text_features


class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens)).cuda()
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class NeSyBase(nn.Module):
    def __init__(self, vid_embed_size, hsize, rnn_enc, text_model,
                 text_feature_extractor='clip', tokenizer=None, context_encoder=None):
        super(NeSyBase, self).__init__()
        # k = 4 (frame + 3 bounding boxes per frame)
        self.vid_ctx_rnn = rnn_enc(4 * vid_embed_size, hsize, bidirectional=True, dropout_p=0, n_layers=1,
                                   rnn_type="lstm")
        self.text_ctx_rnn = rnn_enc(vid_embed_size, hsize, bidirectional=True, dropout_p=0, n_layers=1,
                                    rnn_type="lstm")
        self.text_model = text_model
        self.text_model.eval()
        self.text_feature_extractor = text_feature_extractor
        self.tokenizer = tokenizer

        # context encoding,
        # default=None
        self.context_encoder = context_encoder
        if self.context_encoder == 'mha':  # multi-head attention
            # positional encoding
            self.positional_encode = PositionalEncoding(num_hiddens=2*hsize, dropout=0.5)
            # multi-headed attention
            self.multihead_attn = nn.MultiheadAttention(embed_dim=2*hsize,
                                                        num_heads=10,
                                                        dropout=0.5,
                                                        batch_first=True)
        elif self.context_encoder == 'bilstm':
            self.bilstm = nn.LSTM(input_size=2*hsize,
                                  hidden_size=hsize,
                                  batch_first=True,
                                  bidirectional=True)
        # num_states = 4  # hot, cold, cleaned
        # num_relations = 2  # InReceptacle, Holds, slice
        self.state_query = nn.Sequential(nn.Linear(4 * hsize, hsize),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(hsize, 1))
        self.relation_query = nn.Sequential(nn.Linear(4 * hsize, hsize),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(hsize, 1))
        self.all_sorts = []


    def query(self, query_type, seg_text_feats, vid_feature):
        """
        given segment video features and query features
        output a logit for querying the segment with the text features
        """
        if query_type == 'StateQuery':
            return self.state_query(torch.cat((seg_text_feats, vid_feature), dim=-1))[0]
        elif query_type == 'RelationQuery':
            return self.relation_query(torch.cat((seg_text_feats, vid_feature), dim=-1))[0]


    def process_nodes(self, nodes):
        """
        nodes: nodes of graph in DSL
        returns: list of node args in str format
        """
        pred_args = []
        queries = []
        for node in nodes:
            # 'Step 1 StateQuery(apple,heat)'
            node = re.sub('Step \d+ ', '', node)
            node = re.sub(r"[()]", " ", node).strip().split(" ")
            query_type, pred_arg = node[0], ','.join(node[1:])
            split_text = [pred_arg.split(',')[0], pred_arg.split(',')[-1]]
            pred_args.append(' '.join(split_text))
            queries.append(query_type)
        return pred_args, queries


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
        logits_arr = [[]] * len(all_sorts)  # keeps track of logits matrix (for each cell)
        for ind, sorted_nodes in enumerate(all_sorts):
            nodes = all_sorts[ind]
            pred_args, queries = self.process_nodes(nodes)

            with torch.no_grad():
                segment_text_feats = extract_text_features(hypotheses=pred_args,
                                                           model=self.text_model,
                                                           feature_extractor=self.text_feature_extractor,
                                                           tokenizer=self.tokenizer)
                seg_text_feats, seg_text_lens = segment_text_feats  # [num_nodes, max_tokens=20, 512], [b]

            _, seg_text_feats = self.text_ctx_rnn(seg_text_feats, seg_text_lens)  # [num_nodes, 2*hsize]
            seg_text_feats = seg_text_feats.unsqueeze(0)  # [1, num_nodes, 2*hsize]

            num_nodes = len(sorted_nodes)
            parent_dict[ind] = {k1: {k2: tuple() for k2 in range(num_segments)} for k1 in sorted_nodes}
            # array keeps track of cumulative max logprob for each cell
            arr = torch.full((num_nodes, num_segments), torch.tensor(-100.)).cuda()
            logits_arr[ind] = torch.zeros((num_nodes, num_segments)).cuda()

            # setting the start & end indices of each node on segments
            start_ind = dict(zip(sorted_nodes, np.arange(0, num_nodes, 1)))
            end_ind = dict(zip(sorted_nodes, [num_segments - num_nodes + i for i in range(num_nodes)]))

            # starting outer loop from the last node
            for node_ind, node in zip(np.arange(num_nodes - 1, -1, -1), reversed(sorted_nodes)):
                # starting inner loop from the last segment
                for segment_ind in range(end_ind[node], start_ind[node] - 1, -1):

                    # setting the value of the last column
                    if segment_ind == num_segments - 1:
                        logit = self.query(queries[node_ind],
                                            seg_text_feats[:, node_ind, :],
                                            vid_feature[:, segment_ind, :])
                        arr[node_ind][segment_ind] = F.logsigmoid(logit)
                        logits_arr[ind][node_ind][segment_ind] = logit
                        parent_dict[ind][node][segment_ind] = (segment_ind,)
                        continue

                    logit = self.query(queries[node_ind],
                                       seg_text_feats[:, node_ind, :],
                                       vid_feature[:, segment_ind, :])

                    # setting the values of the last row (except last cell in the row)
                    if node_ind == num_nodes - 1:
                        V_opt_curr = F.logsigmoid(logit)
                        V_opt_next = arr[node_ind][segment_ind + 1]
                        if V_opt_curr >= V_opt_next:
                            arr[node_ind][segment_ind] = V_opt_curr
                            parent_dict[ind][node][segment_ind] =  (segment_ind,)
                        else:
                            arr[node_ind][segment_ind] = V_opt_next
                            parent_dict[ind][node][segment_ind] = \
                                parent_dict[ind][sorted_nodes[node_ind]][segment_ind + 1]

                    # calculating the values of the remaining cells
                    # dp[i][j] = max(query(i,j) + dp[i+1][j], dp[i][j+1])
                    else:
                        # V_opt_curr = F.logsigmoid(logit) + arr[node_ind + 1][segment_ind]  # relaxation added
                        V_opt_curr = F.logsigmoid(logit) + arr[node_ind + 1][segment_ind + 1]  # no relaxation
                        V_opt_next = arr[node_ind][segment_ind + 1]
                        if V_opt_curr >= V_opt_next:
                            arr[node_ind][segment_ind] = V_opt_curr
                            parent_dict[ind][node][segment_ind] = \
                                    (segment_ind,) + parent_dict[ind][sorted_nodes[node_ind + 1]][segment_ind + 1]
                        else:
                            arr[node_ind][segment_ind] = V_opt_next
                            parent_dict[ind][node][segment_ind] = \
                                parent_dict[ind][sorted_nodes[node_ind]][segment_ind + 1]
                    logits_arr[ind][node_ind][segment_ind] = logit

            max_arr[ind] = arr[0][0] / len(sorted_nodes)  # normalizing to account for varying length sequences
        max_sort_ind = torch.tensor(max_arr).argmax()
        # TODO: could be more than one optimum paths
        best_alignment = parent_dict[max_sort_ind][all_sorts[max_sort_ind][0]][0]
        aggregated_logits = torch.tensor(0.).cuda()
        # aggregated_logits = []
        # length normalized aggregation of logits
        for i, j in zip(np.arange(num_nodes), best_alignment):
            aggregated_logits +=  logits_arr[max_sort_ind][i][j] / len(all_sorts[max_sort_ind])
            # aggregated_logits.append(logits_arr[max_sort_ind][i][j])
        return max_sort_ind, max_arr[max_sort_ind], \
               list(zip(all_sorts[max_sort_ind], best_alignment)), aggregated_logits


    def forward(self, vid_feats, graphs, true_labels, task_types, train=True):
        ent_probs = []
        labels = []
        pred_alignments = []
        tasks = []
        # if axis_stats is not None:
        #     axis_stats_batch = []
        for index, (vid_feat, graph, hypothesis, label, task_type) in enumerate(zip(vid_feats, *graphs, true_labels, task_types)):
            # processing the video features
            # each vid_feat is [num_segments, frames_per_segment, 512]
            b, vid_len, _ = vid_feat.shape
            vid_lens = torch.full((b,), vid_len).cuda()
            _, vid_feat = self.vid_ctx_rnn(vid_feat, vid_lens)  # aggregate
            vid_feat = vid_feat.unsqueeze(0)  # [1, num_segments, 512]

            # context encoding into segments
            if self.context_encoder == 'mha':  # multi-head attention
                vid_feat = self.positional_encode(vid_feat)
                # integrating temporal component into each segment encoding
                vid_feat = self.multihead_attn(vid_feat, vid_feat, vid_feat, need_weights=False)[0]
            elif self.context_encoder == 'bilstm':
                vid_feat, (_, _) = self.bilstm(vid_feat)

            # dynamic programming
            try:
                all_sorts = NeSyBase.all_topo_sorts(graph)
                sorted_seq_ind, best_score, best_alignment, aligned_aggregated = self.dp_align(all_sorts, vid_feat)
                pred_alignments.append(best_alignment)
            except:
                print(hypothesis)
                print(GraphEditDistance.nx_to_string(graph))
                continue

            ent_probs.append(torch.sigmoid(aligned_aggregated))
            labels.append(label)
            tasks.append(task_type)
            # if axis_stats is not None:
            #     axis_stats_batch.append(axis_stats[index])

        # if axis_stats is not None:
        #     return torch.stack(ent_probs).view(-1), torch.stack(labels), axis_stats_batch
        if not train:
            return torch.stack(ent_probs).view(-1), torch.stack(labels), pred_alignments, tasks
        return torch.stack(ent_probs).view(-1), torch.stack(labels)
