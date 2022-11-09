import itertools
import re
import torch
import torch.nn as nn
from dataset_utils import tokenize_and_pad
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from transformers import DistilBertModel, DistilBertTokenizer


class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class NeSyBase(nn.Module):
    def __init__(self, text_feature_extractor):
        super(NeSyBase, self).__init__()
        self.text_feature_extractor = text_feature_extractor
        if self.text_feature_extractor == 'bert':
            # distil bert model
            self.text_embed_size = 768
            self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            # transformers use layer norm (and not batch norm) which is local -- no need to sync across all instances
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            # text_model.eval()
        elif self.text_feature_extractor == 'glove':
            self.text_embed_size = 300
            self.tokenizer = get_tokenizer("basic_english")
            self.text_model = GloVe(name='840B', dim=self.text_embed_size)
        else:
            raise NotImplementedError
        self.vid_embed_size = 768

        # positional encoding
        self.positional_encode = PositionalEncoding(num_hiddens=self.vid_embed_size, dropout=0.5)
        # multi-headed attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.vid_embed_size,
                                                    num_heads=8,
                                                    batch_first=True)
        self.num_states = 4  # hot, cold, cleaned, sliced
        self.num_relations = 2  # InReceptacle, Holds
        # TODO: make it deeper?
        self.state_query = nn.Sequential(nn.Linear(2 * self.text_embed_size + self.vid_embed_size, 1),
                                         nn.LogSigmoid())
        self.relation_query = nn.Sequential(nn.Linear(2 * self.text_embed_size + self.vid_embed_size, 1),
                                            nn.LogSigmoid())

    def query(self, segment_ids, nodes, vid_feature):
        """
        given segment_ids and nodes (of a specific level)
        that must query from the segment_ids in the given order,
        output a probability of query(ies)
        here, value function is a log_probability
        """
        log_prob = torch.tensor(0.).cuda()
        for node, segment_id in zip(nodes, segment_ids):
            node = re.sub('Step \d+ ', '', node)
            query_type, pred_args, _ = re.sub(r"[()]", " ", node).split(" ")
            pred_args = pred_args.split(',')
            tokenizer_out = tokenize_and_pad(pred_args, self.tokenizer, self.text_feature_extractor)
            with torch.no_grad():
                text_feats = torch.stack([self.text_model.get_vecs_by_tokens(x).cuda() for x in tokenizer_out])
            if query_type == 'StateQuery':
                log_prob = log_prob + self.state_query(torch.cat((text_feats[:2].contiguous().reshape(-1),
                                                                      vid_feature[segment_id].contiguous())))[0].contiguous()
            elif query_type == 'RelationQuery':
                log_prob = log_prob + self.relation_query(torch.cat((text_feats[:2].contiguous().reshape(-1),
                                                                     vid_feature[segment_id].contiguous())))[0].contiguous()
        # breakpoint()
        return log_prob

    def recurse(self, level_ind, start_ind, levels_start, levels, levels_end, vid_feature):
        # TODO: optimize it by creating a matrix/dict with keys {level_ind, start_ind} for reuse
        """
        generates all permutations within a level based on the start_ind and
        recurses for the max probability based on the current_permutation and
        max_prob of the next level
        """
        # handling corner cases
        if level_ind >= len(levels):
            return torch.tensor(0.).cuda()
        assert start_ind >= levels_start[level_ind]

        end_ind = levels_end[level_ind]
        # -sys.maxsize + 1
        max_val_level = torch.tensor(-100.).cuda()
        nodes = levels[level_ind]  # in that level
        num_nodes = len(nodes)
        all_perm = itertools.permutations(range(start_ind, end_ind+1), num_nodes)
        for perm in all_perm:
            k = max(perm)
            max_val_level = max(max_val_level,
                                self.query(perm, nodes, vid_feature) + self.recurse(level_ind=level_ind + 1,
                                                                                    start_ind=k + 1,
                                                                                    levels_start=levels_start,
                                                                                    levels=levels,
                                                                                    levels_end=levels_end,
                                                                                    vid_feature=vid_feature))
        if max_val_level == torch.tensor(-100.).cuda():
            breakpoint()
        return max_val_level

    def recurse2(self, levels, levels_end, vid_feature):
        """
        optimized recurse: stores values of subproblems
        """
        arr = torch.full((len(levels) + 1, len(vid_feature) + 1), torch.tensor(-100.).cuda()).cuda()
        arr[:, len(vid_feature)] = torch.zeros(len(levels) + 1).cuda()
        arr[len(levels), :] = torch.zeros(len(vid_feature) + 1).cuda()

        for level_ind in range(len(levels) - 1, -1, -1):
            nodes = levels[level_ind]  # in that level
            num_nodes = len(nodes)
            end_ind = levels_end[level_ind]
            for segment_ind in range(len(vid_feature) - 1, -1, -1):
                start_ind = segment_ind
                all_perm = itertools.permutations(range(start_ind, end_ind + 1), num_nodes)
                for perm in all_perm:
                    k = max(perm)
                    if level_ind <= len(levels) - 1:
                        # assert k+1 >= levels_start[level_ind+1]
                        arr[level_ind][start_ind] = max(arr[level_ind][start_ind],
                                                        self.query(perm, nodes, vid_feature) +
                                                        arr[level_ind + 1, k + 1])

        return arr[0][0].contiguous()


    @staticmethod
    def group_nodes_into_levels(graph, num_segments):
        nodes = graph.nodes
        # TODO: must uncomment this
        # assert len(nodes) <= num_segments  # N <= S

        levels = {k: None for k in nodes}
        edges = set(graph.edges)
        adj = {k: [] for k in nodes}
        for edge in edges:
            src_node, dest_node, _ = edge
            adj[src_node].append(dest_node)

        src_nodes = set(nodes) - set().union(*adj.values())
        # src node
        for node in src_nodes:
            levels[node] = 0
        all_nodes = src_nodes.copy()
        while len(all_nodes) > 0:
            node = all_nodes.pop()
            for adj_node in adj[node]:
                levels[adj_node] = levels[node] + 1
                all_nodes.add(adj_node)

        levels = {v: [i for i in levels.keys() if levels[i] == v] for k, v in levels.items()}
        # levels_start, levels_end = [], []
        for ind in range(len(levels)):
            if ind == 0:
                levels_start = [0]
                continue
            levels_start.append(levels_start[-1] + len(list(levels.values())[ind - 1]))
        for ind in range(len(levels) - 1, -1, -1):
            if ind == len(levels) - 1:
                levels_end = [num_segments-1]
                continue
            levels_end.insert(0, levels_end[0] - len(list(levels.values())[ind + 1]))
        # levels_start = [np.array(levels.values()[:ind]).sum() for ind in range(len(levels))]
        return levels, levels_start, levels_end

    def forward(self, vid_feats, graphs):
        out_vals = []  # log probs
        vid_features, vid_lens = vid_feats
        vid_features = self.positional_encode(vid_features)
        # integrating temporal component into each segment encoding
        vid_features = self.multihead_attn(vid_features, vid_features, vid_features, need_weights=False)[0]
        for vid_feature, vid_len, graph in zip(vid_features, vid_lens, graphs):
            num_segments = vid_len.item()  # rest is all padding
            levels, levels_start, levels_end = NeSyBase.group_nodes_into_levels(graph, num_segments)
            # max_val = self.recurse(0, 0, levels_start, levels, levels_end, vid_feature)
            max_val = self.recurse2(levels, levels_end, vid_feature)
            out_vals.append(max_val)
        return torch.stack(out_vals).reshape(-1)  # log probs
