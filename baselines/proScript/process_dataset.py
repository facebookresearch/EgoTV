import os
import json
import pydot
from tqdm import tqdm
from nltk.tokenize import word_tokenize


def get_output(goal, pickup, recep=None):
    # nodes
    graph = pydot.Dot("graph", graph_type="digraph")
    nodes = goal.replace('then', 'and').split('_and_')[:]
    node_strs = []
    for ind, node in enumerate(nodes):
        if node == 'place':
            node_str = "Step {} {} {} in {}".format(ind + 1, node, pickup, recep)
        else:
            node_str = "Step {} {} {}".format(ind+1, node, pickup)
        graph.add_node(pydot.Node(node_str))
        node_strs.append(node_str)

    # edges
    order_list = goal.split('_then_')
    for subgoals1, subgoals2 in zip(order_list[:-1], order_list[1:]):
        if 'and' in subgoals1:
            subgoals1 = subgoals1.split('_and_')
        else:
            subgoals1 = [subgoals1]

        if 'and' in subgoals2:
            subgoals2 = subgoals2.split('_and_')
        else:
            subgoals2 = [subgoals2]

        for s1 in subgoals1:
            s1_str = node_strs[nodes.index(s1)]
            for s2 in subgoals2:
                s2_str =  node_strs[nodes.index(s2)]
                graph.add_edge(pydot.Edge(src=s1_str, dst=s2_str))

    # graph.write_png("output.png")
    return graph.to_string().replace('\n','').replace('digraph graph {', '').replace('}', '')
    # pydot.graph_from_dot_data(graph.to_string().replace(';', ';\n').replace('{', '{\n').replace('}', '}\n'))[0]
    # proscript_data_path.write('\n')


def proScript_process(dir, data_filename):
    source_len_max = 0
    target_len_max = 0
    data_filename = open(data_filename, "w")
    goals_count = [len(goals) for _, goals, _ in os.walk(dir)][0]

    with tqdm(total=goals_count) as pbar:
        for root, goals, _ in os.walk(dir):
            # breakpoint()
            for goal in goals:
                if goal == 'fails':
                    continue
                pbar.update(1)
                for traj_root, trials, _ in os.walk(os.path.join(root, goal)):
                    for trial in trials:
                        if trial.count('-') == 3:
                            pickup, _, receptacle, _ = trial.split('-')
                            pickup = pickup.replace("sliced", "")
                            for file_path, _dirs, _ in os.walk(os.path.join(traj_root, trial)):
                                for _d in _dirs:
                                    for trial_path, _, _files in os.walk(os.path.join(traj_root, trial, _d)):
                                        if 'video.mp4' in _files:
                                            json_path = os.path.join(trial_path, 'traj_data.json')
                                            traj = json.load(open(json_path, 'r'))
                                            graph_str = get_output(goal, pickup.lower(), receptacle.lower())
                                            data_filename.write("%s\t%s\n" % (traj['template']['pos'], graph_str))
                                            source_len_max = max(source_len_max,
                                                                 len(word_tokenize(traj['template']['pos'])))
                                            target_len_max = max(target_len_max, len(word_tokenize(graph_str)))
                                        break  # only examine top level
                                break  # only examine top level
            break  # only examine top level
    data_filename.close()
    return source_len_max, target_len_max


if __name__=="__main__":
    dir = '../../alfred/gen/dataset/context_goal_composition'
    proScript_process(dir, data_filename="proscript_data.tsv")