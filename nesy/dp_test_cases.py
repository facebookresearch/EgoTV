import os
import sys
os.environ['BASELINES'] = '/mnt/c/Users/rishihazra/PycharmProjects/VisionLangaugeGrounding/baselines'
sys.path.append(os.environ['BASELINES'])
from proScript.proscript_utils import GraphEditDistance
from nesy_model import NeSyBase
import itertools
import numpy as np


def test_level_grouping():
    """
    test NeSyBase.group_nodes_into_levels()
    """
    ged = GraphEditDistance()
    # Case 1 a potato is cleaned and sliced, then placed in a plate
    tf_out = ['"Step 1 StateQuery(potato,sliced)";'
              '"Step 2 StateQuery(potato,clean)";'
              '"Step 3 RelationQuery(potato,plate,inReceptacle)";'
              '"Step 1 StateQuery(potato,sliced)" -> "Step 3 RelationQuery(potato,plate,inReceptacle)";'
              '"Step 2 StateQuery(potato,clean)" -> "Step 3 RelationQuery(potato,plate,inReceptacle)";']
    graph = ged.pydot_to_nx(tf_out[0])
    num_segments = 5
    levels, levels_start, levels_end = NeSyBase.group_nodes_into_levels(graph=graph, num_segments=num_segments)
    bin_assignment = np.array([0, 0.2, 0, 0.6, 0.8])
    # bin_assignment = np.zeros(num_segments)
    # bin_assignment[rand_indices] = 1
    sol = recurse2(levels, levels_end, bin_assignment, levels_start)
    if sol == bin_assignment.sum():
        print("Test Case 1: PASSED")
    else:
        print("Test Case 1: FAILED")vdjnbehtrtcllhugjfgknetuniefunbe

    # Case 2 tomato is cleaned and sliced and heated
    tf_out = ['"Step 1 StateQuery(tomato,sliced)";'
              '"Step 2 StateQuery(tomato,clean)";'
              '"Step 3 StateQuery(tomato,hot)";']
    graph = ged.pydot_to_nx(tf_out[0])
    levels, levels_start, levels_end = NeSyBase.group_nodes_into_levels(graph=graph, num_segments=num_segments)
    rand_indices = np.array([0, 1, 4])
    bin_assignment = np.zeros(num_segments)
    bin_assignment[rand_indices] = 1
    if recurse2(levels, levels_end, bin_assignment, levels_start) == len(graph.nodes):
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
    levels, levels_start, levels_end = NeSyBase.group_nodes_into_levels(graph=graph, num_segments=num_segments)
    rand_indices = np.array([0, 1, 3, 5])
    bin_assignment = np.zeros(num_segments)
    bin_assignment[rand_indices] = 1
    if recurse2(levels, levels_end, bin_assignment, levels_start) == len(graph.nodes):
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
    levels, levels_start, levels_end = NeSyBase.group_nodes_into_levels(graph=graph, num_segments=num_segments)
    rand_indices = np.array([2, 4, 5, 7])
    bin_assignment = np.zeros(num_segments)
    bin_assignment[rand_indices] = 1
    if recurse2(levels, levels_end, bin_assignment, levels_start) == len(graph.nodes):
        print("Test Case 4: PASSED")
    else:
        print("Test Case 4: FAILED")

    # Case 5 a potato is heated, then cleaned and sliced, then placed in a plate
    tf_out = ['"Step 1 StateQuery(potato,hot)";']
    graph = ged.pydot_to_nx(tf_out[0])
    num_segments = 5
    levels, levels_start, levels_end = NeSyBase.group_nodes_into_levels(graph=graph, num_segments=num_segments)
    rand_indices = np.array([1, 2, 3, 4])
    bin_assignment = np.zeros(num_segments)
    bin_assignment[rand_indices] = 1
    if recurse2(levels, levels_end, bin_assignment, levels_start) == len(graph.nodes):
        print("Test Case 5: PASSED")
    else:
        print("Test Case 5: FAILED")


def query(perm, bin_assignment):
    net = 0.
    for perm_ind in perm:
        net += bin_assignment[perm_ind]
    return net


def recurse(level_ind, start_ind, levels_start, levels, levels_end, bin_assignment):
    # TODO: optimize it by creating a matrix/dict with keys {level_ind, start_ind} for reuse
    """
    generates all permutations within a level based on the start_ind and
    recurses for the max probability based on the current_permutation and
    max_prob of the next level
    """
    # handling corner cases
    if level_ind >= len(levels):
        return 0.
    assert start_ind >= levels_start[level_ind]

    end_ind = levels_end[level_ind]
    # -sys.maxsize + 1
    max_val_level = -100.
    nodes = levels[level_ind]  # in that level
    num_nodes = len(nodes)
    all_perm = itertools.permutations(range(start_ind, end_ind+1), num_nodes)
    for perm in all_perm:
        k = max(perm)
        # bin_ind = [x for x in np.where(np.array(bin_assignment) == 1)[0] if x >= start_ind and x < end_ind+1]
        max_val_level = max(max_val_level,
                            query(perm, bin_assignment) + recurse(level_ind=level_ind + 1,
                                                                  start_ind=k + 1,
                                                                  levels_start=levels_start,
                                                                  levels=levels,
                                                                  levels_end=levels_end,
                                                                  bin_assignment=bin_assignment))
    if max_val_level == -100.:
        print('What ... Why ... How ... ?')
    return max_val_level


def recurse2(levels, levels_end, bin_assignment, levels_start):
    # TODO: optimize it by creating a matrix/dict with keys {level_ind, start_ind} for reuse
    """
    optimized recurse: stores values of sub-problems
    """
    path_opt = [[None]] * len(levels)
    arr = np.zeros((len(levels)+1, len(bin_assignment)+1))
    for level_ind in range(len(levels)-1, -1, -1):
        num_nodes = len(levels[level_ind])
        end_ind = levels_end[level_ind]
        for segment_ind in range(len(bin_assignment)-1, levels_start[level_ind]-1, -1):
            start_ind = segment_ind
            all_perm = itertools.permutations(range(start_ind, end_ind + 1), num_nodes)
            for perm in all_perm:
                k = max(perm)
                if level_ind <= len(levels) - 1:
                    V_opt_next = arr[level_ind + 1, k + 1]
                    V_opt = query(perm, bin_assignment) + V_opt_next
                    if arr[level_ind][start_ind] < V_opt:
                        path_opt[level_ind] = perm
                        arr[level_ind][start_ind] = V_opt
    return arr[0][0]


if __name__ == '__main__':
    test_level_grouping()  # passed all test cases
