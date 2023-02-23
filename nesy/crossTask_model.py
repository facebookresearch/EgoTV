import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extraction import extract_text_features


class NeSyBase(nn.Module):
    def __init__(self, vid_embed_size, hsize, rnn_enc, text_model):
        super(NeSyBase, self).__init__()
        # TODO: generalize to 'k' bounding boxes instead of 2
        self.vid_ctx_rnn = rnn_enc(vid_embed_size, hsize, bidirectional=True, dropout_p=0, n_layers=1,
                                   rnn_type="lstm")
        self.text_ctx_rnn = rnn_enc(vid_embed_size, hsize, bidirectional=True, dropout_p=0, n_layers=1,
                                    rnn_type="lstm")
        self.text_model = text_model
        self.action_query = nn.Sequential(nn.Linear(4 * hsize, hsize),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(hsize, 1))
        self.all_sorts = []

    def query(self, seg_text_feats, vid_feature):
        """
        given segment video features and query features
        output a logit for querying the segment with the text features
        """
        return self.action_query(torch.cat((seg_text_feats, vid_feature), dim=-1))[0]


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
            # pred_args, queries = self.process_nodes(nodes)
            with torch.no_grad():
                segment_text_feats = extract_text_features(nodes, self.text_model, 'clip', tokenizer=None)
            seg_text_feats, seg_text_lens = segment_text_feats
            _, seg_text_feats = self.text_ctx_rnn(seg_text_feats, seg_text_lens)
            seg_text_feats = seg_text_feats.unsqueeze(0)  # [1, num_nodes, 512]

            num_nodes = len(sorted_nodes)
            parent_dict[ind] = {k1: {k2: tuple() for k2 in range(num_segments)} for k1 in sorted_nodes}
            # array keeps track of cumulative max logprob for each cell
            arr = torch.full((num_nodes, num_segments), torch.tensor(-100.)).cuda()
            logits_arr[ind] = torch.zeros((num_nodes, num_segments)).cuda()
            start_ind = dict(zip(sorted_nodes, np.arange(0, num_nodes, 1)))
            end_ind = dict(zip(sorted_nodes, [num_segments - num_nodes + i for i in range(num_nodes)]))
            for node_ind, node in zip(np.arange(num_nodes - 1, -1, -1), reversed(sorted_nodes)):
                for segment_ind in range(end_ind[node], start_ind[node] - 1, -1):
                    # TODO: relax this to arr[node_ind+1][segment_ind]

                    if segment_ind == num_segments - 1:
                        logit = self.query(seg_text_feats[:, node_ind, :],
                                           vid_feature[:, segment_ind, :])
                        arr[node_ind][segment_ind] =  F.logsigmoid(logit)
                        logits_arr[ind][node_ind][segment_ind] = logit
                        parent_dict[ind][node][segment_ind] = (segment_ind,)
                        continue

                    logit = self.query(seg_text_feats[:, node_ind, :],
                                       vid_feature[:, segment_ind, :])
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
                    else:
                        V_opt_curr = F.logsigmoid(logit) + arr[node_ind + 1][segment_ind]  # relaxation added
                        # V_opt_curr = F.logsigmoid(logit) + arr[node_ind + 1][segment_ind + 1]  # no relaxation
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

            max_arr[ind] = arr[0][0]
        max_sort_ind = torch.tensor(max_arr).argmax()
        # TODO: could be more than one optimum paths
        best_alignment = parent_dict[max_sort_ind][all_sorts[max_sort_ind][0]][0]
        aggregated_logits = torch.tensor(0.).cuda()
        for i, j in zip(np.arange(num_nodes), best_alignment):
            aggregated_logits +=  logits_arr[max_sort_ind][i][j]
        return max_sort_ind, max_arr[max_sort_ind], \
               list(zip(all_sorts[max_sort_ind], best_alignment)), aggregated_logits


    def forward(self, vid_feats, all_sorts_batch, true_labels, train=True):
        ent_probs = []
        labels = []
        pred_alignments = []
        for vid_feat, all_sorts, label in zip(vid_feats, all_sorts_batch, true_labels):
            # processing the video features
            # each vid_feat is [num_segments, frames_per_segment, 512]
            b, vid_len, _ = vid_feat.shape
            vid_lens = torch.full((b,), vid_len).cuda()
            _, vid_feat = self.vid_ctx_rnn(vid_feat, vid_lens)  # aggregate
            vid_feat = vid_feat.unsqueeze(0)  # [1, num_segments, 512]

            # dynamic programming
            # assuming we already have all sorted sequences
            sorted_seq_ind, best_score, best_alignment, aligned_aggregated = self.dp_align(all_sorts, vid_feat)
            pred_alignments.append(best_alignment)

            ent_probs.append(torch.sigmoid(aligned_aggregated))
            labels.append(label)

        if not train:
            return torch.stack(ent_probs).view(-1), torch.stack(labels), pred_alignments
        return torch.stack(ent_probs).view(-1), torch.stack(labels)
