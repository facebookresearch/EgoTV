import os
import sys
os.environ['BASELINES'] = '/mnt/c/Users/rishihazra/PycharmProjects/VisionLangaugeGrounding/baselines'
sys.path.append(os.environ['BASELINES'])
from proScript.utils import GraphEditDistance
from nesy_model import NeSyBase
import numpy as np


def test_level_grouping():
    """
    test NeSyBase.group_nodes_into_levels()
    """
    ged = GraphEditDistance()
    # Case 1 a potato is cleaned and sliced, then placed in a plate
    tf_out = ['"Step 1 StateQuery(potato,sliced)";']
    graph = ged.pydot_to_nx(tf_out[0])
    num_segments = 5
    # sorted_nodes = NeSyBase.topo_sort(graph)
    all_sorts = NeSyBase.all_topo_sorts(graph)
    bin_assignment = np.zeros((len(all_sorts), len(graph.nodes), num_segments))
    bin_assignment[0, 0, :] = np.array([0.4, 0.5, 0.6, 0.0, 0.0])
    bin_assignment[0, 1, :] = np.array([0.0, 0.2, 0.4, 0.6, 0.0])
    bin_assignment[0, 2, :] = np.array([0.0, 0.0, 0.3, 0.8, 0.6])

    bin_assignment[1, 0, :] = np.array([0.2, 0.4, 0.5, 0.0, 0.0])
    bin_assignment[1, 1, :] = np.array([0.0, 0.2, 0.4, 0.9, 0.0])
    bin_assignment[1, 2, :] = np.array([0.0, 0.0, 0.3, 0.8, 0.3])
    sorted_seq_ind, best_score, best_alignment = recurse(all_sorts, bin_assignment)
    if round(best_score, 2) == 1.8 and best_alignment == (2, 3, 4) and len(all_sorts) == 2 and sorted_seq_ind == 0:
        print("Test Case 1: PASSED")
    else:
        print("Test Case 1: FAILED")

    # Case 2 tomato is cleaned and sliced and heated
    tf_out = ['"Step 1 StateQuery(tomato,sliced)";'
              '"Step 2 StateQuery(tomato,clean)";'
              '"Step 3 StateQuery(tomato,hot)";']
    graph = ged.pydot_to_nx(tf_out[0])
    all_sorts = NeSyBase.all_topo_sorts(graph)
    bin_assignment = np.zeros((len(all_sorts), len(graph.nodes), num_segments))
    bin_assignment[0, 0, :] = np.array([0.4, 0.5, 0.6, 0.0, 0.0])
    bin_assignment[0, 1, :] = np.array([0.0, 0.2, 0.4, 0.6, 0.0])
    bin_assignment[0, 2, :] = np.array([0.0, 0.0, 0.3, 0.8, 0.6])

    bin_assignment[1, 0, :] = np.array([0.2, 0.4, 0.5, 0.0, 0.0])
    bin_assignment[1, 1, :] = np.array([0.0, 0.2, 0.4, 0.9, 0.0])
    bin_assignment[1, 2, :] = np.array([0.0, 0.0, 0.3, 0.8, 0.3])

    bin_assignment[2, 0, :] = np.array([0.9, 0.4, 0.5, 0.0, 0.0])
    bin_assignment[2, 1, :] = np.array([0.0, 0.9, 0.4, 0.9, 0.0])
    bin_assignment[2, 2, :] = np.array([0.0, 0.0, 0.1, 0.8, 0.3])

    sorted_seq_ind, best_score, best_alignment = recurse(all_sorts, bin_assignment)
    if best_score == 2.6 and len(all_sorts) == 6 and best_alignment == (0, 1, 3) and sorted_seq_ind == 2:
        print("Test Case 2: PASSED")
    else:
        print("Test Case 2: FAILED")

    # Case 3 apple is heated, then cleaned, sliced and cooled
    tf_out = ['"Step 1 StateQuery(apple,hot)";'
              '"Step 2 StateQuery(apple,sliced)";'
              '"Step 3 StateQuery(apple,cold)";'
              '"Step 4 StateQuery(apple,clean)";'
              '"Step 1 StateQuery(apple,hot)" -> "Step 2 StateQuery(apple,sliced)";'
              '"Step 1 StateQuery(apple,hot)" -> "Step 3 StateQuery(apple,cold)";'
              '"Step 1 StateQuery(apple,hot)" -> "Step 4 StateQuery(apple,clean)";']
    graph = ged.pydot_to_nx(tf_out[0])
    num_segments = 6
    all_sorts = NeSyBase.all_topo_sorts(graph)
    bin_assignment = np.zeros((len(all_sorts), len(graph.nodes), num_segments))
    bin_assignment[0, 0, :] = np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0])
    bin_assignment[0, 1, :] = np.array([0.0, 0.2, 0.4, 0.6, 0.0, 0.0])
    bin_assignment[0, 2, :] = np.array([0.0, 0.0, 0.3, 0.8, 0.6, 0.0])
    bin_assignment[0, 3, :] = np.array([0.0, 0.0, 0.0, 0.8, 0.6, 0.1])

    bin_assignment[1, 0, :] = np.array([0.2, 0.4, 0.5, 0.0, 0.0, 0.0])
    bin_assignment[1, 1, :] = np.array([0.0, 0.2, 0.4, 0.9, 0.0, 0.0])
    bin_assignment[1, 2, :] = np.array([0.0, 0.0, 0.3, 0.8, 0.3, 0.0])
    bin_assignment[1, 3, :] = np.array([0.0, 0.0, 0.0, 0.8, 0.6, 0.2])

    bin_assignment[2, 0, :] = np.array([0.1, 0.4, 0.5, 0.0, 0.0, 0.0])
    bin_assignment[2, 1, :] = np.array([0.0, 0.9, 0.4, 0.9, 0.0, 0.0])
    bin_assignment[2, 2, :] = np.array([0.0, 0.0, 0.1, 0.8, 0.3, 0.0])
    bin_assignment[2, 3, :] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.3])
    sorted_seq_ind, best_score, best_alignment = recurse(all_sorts, bin_assignment)
    if best_score == 2.3 and len(all_sorts) == 6 and best_alignment == (1, 2, 3, 4) and sorted_seq_ind == 0:
        print("Test Case 3: PASSED")
    else:
        print("Test Case 3: FAILED")

    # Case 4 a potato is heated, then cleaned and sliced, then placed in a plate
    tf_out = ['"Step 1 StateQuery(potato,hot)";'
              '"Step 2 StateQuery(potato,clean)";'
              '"Step 3 StateQuery(potato,sliced)";'
              '"Step 4 RelationQuery(potato,plate,inReceptacle)";'
              '"Step 1 StateQuery(potato,hot)" -> "Step 2 StateQuery(potato,clean)";'
              '"Step 1 StateQuery(potato,hot)" -> "Step 3 StateQuery(potato,sliced)";'
              '"Step 2 StateQuery(potato,clean)" -> "Step 4 RelationQuery(potato,plate,inReceptacle)";'
              '"Step 3 StateQuery(potato,sliced)" -> "Step 4 RelationQuery(potato,plate,inReceptacle)";']
    graph = ged.pydot_to_nx(tf_out[0])
    num_segments = 8
    all_sorts = NeSyBase.all_topo_sorts(graph)
    bin_assignment = np.zeros((len(all_sorts), len(graph.nodes), num_segments))
    bin_assignment[0, 0, :] = np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.1, 0.0, 0.0])
    bin_assignment[0, 1, :] = np.array([0.0, 0.2, 0.4, 0.6, 0.0, 0.1, 0.0, 0.0])
    bin_assignment[0, 2, :] = np.array([0.0, 0.0, 0.3, 0.8, 0.6, 0.1, 0.0, 0.0])
    bin_assignment[0, 3, :] = np.array([0.0, 0.0, 0.0, 0.8, 0.6, 0.1, 0.7, 0.0])

    bin_assignment[1, 0, :] = np.array([0.2, 0.4, 0.5, 0.0, 0.0, 0.2, 0.0, 0.0])
    bin_assignment[1, 1, :] = np.array([0.0, 0.2, 0.4, 0.9, 0.0, 0.2, 0.0, 0.0])
    bin_assignment[1, 2, :] = np.array([0.0, 0.0, 0.3, 0.8, 0.3, 0.2, 0.0, 0.0])
    bin_assignment[1, 3, :] = np.array([0.0, 0.0, 0.0, 0.8, 0.6, 0.1, 0.0, 0.6])

    sorted_seq_ind, best_score, best_alignment = recurse(all_sorts, bin_assignment)
    if best_score == 2.5  and len(all_sorts) == 2 and best_alignment == (2, 3, 4, 6) and sorted_seq_ind == 0:
        print("Test Case 4: PASSED")
    else:
        print("Test Case 4: FAILED")

    # Case 5 a potato is heated, then cleaned and sliced, then placed in a plate
    tf_out = ['"Step 1 StateQuery(potato,hot)";']
    graph = ged.pydot_to_nx(tf_out[0])
    num_segments = 5
    all_sorts = NeSyBase.all_topo_sorts(graph)
    bin_assignment = np.zeros((len(all_sorts), len(graph.nodes), num_segments))
    bin_assignment[0, 0, :] = np.array([0.4, 0.5, 0.7, 0.0, 0.0])

    sorted_seq_ind, best_score, best_alignment = recurse(all_sorts, bin_assignment)
    if best_score == 0.7 and len(all_sorts) == 1 and best_alignment == 2 and sorted_seq_ind == 0:
        print("Test Case 5: PASSED")
    else:
        print("Test Case 5: FAILED")


def query(node_ind, segment_ind, bin_assignment):
    return bin_assignment[node_ind, segment_ind]


def recurse(all_sorts, bin_assignment):
    """
    optimized recurse: stores values of sub-problems
    """
    max_arr = [[]] * len(all_sorts)  # keep track of max for each sorted sequence
    parent_dict = [[]] * len(all_sorts)  # keeps track of the best path for each sorted sequence
    num_segments = len(bin_assignment[0, 0])
    for ind, sorted_nodes in enumerate(all_sorts):
        num_nodes = len(sorted_nodes)
        parent_dict[ind] = {k1: {k2 : tuple() for k2 in range(num_segments)} for k1 in sorted_nodes}
        arr = np.zeros((num_nodes+1, num_segments+1))  # array keeps track of cumulative max for each cell
        start_ind = dict(zip(sorted_nodes, np.arange(0, num_nodes, 1)))
        end_ind = dict(zip(sorted_nodes, [num_segments - num_nodes + i  for i in range(num_nodes)]))
        for node_ind, node in zip(np.arange(num_nodes-1, -1, -1), reversed(sorted_nodes)):
            for segment_ind in range(end_ind[node], start_ind[node]-1, -1):
                # TODO: relax this to arr[node_ind+1][segment_ind]
                V_opt_curr = query(node_ind, segment_ind, bin_assignment[ind]) + \
                             arr[node_ind+1][segment_ind+1]
                V_opt_next = arr[node_ind][segment_ind+1]
                if V_opt_curr >= V_opt_next:
                    arr[node_ind][segment_ind] = V_opt_curr
                    if node_ind == num_nodes - 1:
                        parent_dict[ind][node][segment_ind] = (segment_ind,)
                    else:
                        parent_dict[ind][node][segment_ind] = \
                            (segment_ind,) + parent_dict[ind][sorted_nodes[node_ind + 1]][segment_ind+1]
                else:
                    arr[node_ind][segment_ind] = V_opt_next
                    parent_dict[ind][node][segment_ind] = parent_dict[ind][sorted_nodes[node_ind]][segment_ind+1]
                # else:
                #     arr[node_ind][segment_ind] = V_opt_curr
                #     parent_dict[ind][node][segment_ind] = \
                #         ((segment_ind,) + parent_dict[ind][sorted_nodes[node + 1]][segment_ind],
                #          parent_dict[ind][sorted_nodes[node + 1]][segment_ind+1])
                #     # parent_dict[ind][node] = parent_dict[ind][sorted_nodes[node + 1]]
        max_arr[ind] = arr[0][0]
    max_sort_ind = np.array(max_arr).argmax()
    #TODO: could be more than one optimum paths
    return max_sort_ind, max_arr[max_sort_ind], parent_dict[max_sort_ind][all_sorts[max_sort_ind][0]][0]


if __name__ == '__main__':
    test_level_grouping()  # passed all test cases
