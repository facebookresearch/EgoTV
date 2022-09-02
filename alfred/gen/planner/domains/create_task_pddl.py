import itertools

all_lines = open('all_goals.pddl').readlines()

# 'stack_then_place', 'place_then_stack',
all_tasks = ['clean_then_cool_then_place',
             'clean_then_cool_then_slice',
             'clean_then_heat_then_place',
             'clean_then_heat_then_slice',
             'clean_then_slice_then_cool',
             'clean_then_slice_then_heat',
             'cool_then_clean_then_place',
             'cool_then_clean_then_slice',
             'cool_then_slice_then_clean',
             'heat_then_clean_then_place',
             'heat_then_clean_then_slice',
             'heat_then_slice_then_clean',
             'slice_then_clean_then_cool',
             'slice_then_clean_then_heat',
             'slice_then_clean_then_place',
             'slice_then_cool_then_clean',
             'slice_then_cool_then_place',
             'slice_then_heat_then_clean',
             'slice_then_heat_then_place',
             'clean_and_cool_then_place',
             'clean_and_cool_then_slice',
             'clean_and_heat_then_place',
             'clean_and_heat_then_slice',
             'clean_and_slice_then_cool',
             'clean_and_slice_then_heat',
             'clean_then_cool_and_slice',
             'clean_then_heat_and_slice',
             'clean_then_slice_and_cool',
             'clean_then_slice_and_heat',
             'cool_and_clean_then_place',
             'cool_and_clean_then_slice',
             'cool_and_slice_then_clean',
             'cool_then_clean_and_slice',
             'cool_then_slice_and_clean',
             'heat_and_clean_then_place',
             'heat_and_clean_then_slice',
             'heat_and_slice_then_clean',
             'heat_then_clean_and_slice',
             'heat_then_slice_and_clean',
             'slice_and_clean_then_cool',
             'slice_and_clean_then_heat',
             'slice_and_clean_then_place',
             'slice_and_cool_then_clean',
             'slice_and_cool_then_place',
             'slice_and_heat_then_clean',
             'slice_and_heat_then_place',
             'slice_then_clean_and_cool',
             'slice_then_clean_and_heat',
             'slice_then_cool_and_clean',
             'slice_then_heat_and_clean']


all_subgoals = ['cool', 'slice', 'heat', 'clean', 'place']
unique_combinations = ['_and_'.join(combination) for combination in itertools.combinations(all_subgoals, 3)
                            if ('place' not in combination[:2]) and ('heat' not in combination or 'cool' not in combination)]
print(unique_combinations)

for task in all_tasks:
    root_task = task.replace('then', 'and')
    subgoals = root_task.split('_and_')
    all_permutations = ['_and_'.join(x) for x in itertools.permutations(subgoals)]
    root_task = set(unique_combinations).intersection(set(all_permutations))
    # for permutation in all_permutations:
    #     print("gdict['{}']['pddl'] = gdict['{}']['pddl']".format(task, '_and_'.join(permutation)))
    print("gdict['{}']['pddl'] = gdict['{}']['pddl']".format(task, list(root_task)[0]))
    # print('\n')

subgoal_to_str_map = {'cool': 'CoolObject', 'heat': 'HeatObject', 'slice': 'SliceObject',
                      'clean': 'CleanObject', 'place': 'PutObjectInReceptacle1'}

# to augment preconditions to the current subgoal based on the previous subgoal
subgoal_precondition_augment = {'cool': ['(isCool ?o)', '(coolable ?o)'], 'heat': ['(isHot ?o)', '(heatable ?o)'],
                                'clean': ['(isClean ?o)', '(cleanable ?o)'], 'slice': ['(isSliced ?o)', '(sliceable ?o)']}

for task in all_tasks:
    savefile = 'task_' + task + '.pddl'
    with open(savefile, 'w') as outfile:

        if 'and' not in task:
            seq_of_subgoals = task.split('_then_')
            inside_subgoal_definition = False

            for line in all_lines:
                outfile.write(line)

                if subgoal_to_str_map[seq_of_subgoals[1]] in line:
                    curr_subgoal = seq_of_subgoals[1]
                    prev_subgoal = seq_of_subgoals[0]
                    inside_subgoal_definition = True
                elif subgoal_to_str_map[seq_of_subgoals[2]] in line:
                    curr_subgoal = seq_of_subgoals[2]
                    prev_subgoal = seq_of_subgoals[1]
                    inside_subgoal_definition = True

                if inside_subgoal_definition:
                    if 'precondition' in line:
                        for augment_condition in subgoal_precondition_augment[prev_subgoal]:
                            augmented_str = '            ' + augment_condition + '\n'
                            outfile.write(augmented_str)

                    if 'effect' in line:
                        inside_subgoal_definition = False
        else:
            first_part, second_part = task.split('_then_')
            inside_subgoal_definition = False

            if 'and' in first_part:
                prev_subgoals = first_part.split('_and_')
                curr_subgoal = second_part

                for line in all_lines:
                    outfile.write(line)

                    if subgoal_to_str_map[curr_subgoal] in line:
                        inside_subgoal_definition = True

                    if inside_subgoal_definition:
                        if 'precondition' in line:
                            for prev_subgoal in prev_subgoals:
                                for augment_condition in subgoal_precondition_augment[prev_subgoal]:
                                    augmented_str = '            ' + augment_condition + '\n'
                                    outfile.write(augmented_str)

                        if 'effect' in line:
                            inside_subgoal_definition = False
            else:
                prev_subgoal = first_part
                curr_subgoals = second_part.split('_and_')

                for line in all_lines:
                    outfile.write(line)

                    for curr_subgoal in curr_subgoals:
                        if subgoal_to_str_map[curr_subgoal] in line:
                            inside_subgoal_definition = True

                    if inside_subgoal_definition:
                        if 'precondition' in line:
                            for augment_condition in subgoal_precondition_augment[prev_subgoal]:
                                augmented_str = '            ' + augment_condition + '\n'
                                outfile.write(augmented_str)

                        if 'effect' in line:
                            inside_subgoal_definition = False




