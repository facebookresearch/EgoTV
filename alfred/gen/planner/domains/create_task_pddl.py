import itertools

all_lines = open('all_goals.pddl').readlines()

all_tasks = ['clean_then_cool',
             'clean_then_heat',
             'clean_then_place',
             'clean_then_slice',
             'cool_then_clean',
             'cool_then_place',
             'cool_then_slice',
             'heat_then_clean',
             'heat_then_place',
             'heat_then_slice',
             'slice_then_clean',
             'slice_then_cool',
             'slice_then_heat',
             'slice_then_place',
             'clean_then_cool_then_place',
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
             'cool_and_slice_then_clean',
             'cool_then_clean_and_slice',
             'heat_and_slice_then_clean',
             'heat_then_clean_and_slice',
             'slice_and_clean_then_place',
             'slice_and_cool_then_place',
             'slice_and_heat_then_place',
             'slice_then_clean_and_cool',
             'slice_then_clean_and_heat',
             'clean_then_cool_and_place',
             'clean_then_heat_and_place',
             'clean_then_slice_and_place',
             'slice_then_cool_and_place',
             'slice_then_heat_and_place',
             'slice_then_clean_and_place',
             'heat_then_clean_and_place',
             'heat_then_slice_and_place',
             'cool_then_clean_and_place',
             'cool_then_slice_and_place']

all_subgoals = ['cool', 'slice', 'heat', 'clean', 'place']
unique_combinations = ['_and_'.join(combination) for combination in itertools.combinations(all_subgoals, 3)
                       if ('place' not in combination[:2]) and ('heat' not in combination or 'cool' not in combination)]
# print(unique_combinations)

for task in all_tasks:
    root_task = task.replace('then', 'and')
    subgoals = root_task.split('_and_')
    all_permutations = ['_and_'.join(x) for x in itertools.permutations(subgoals)]
    root_task = set(unique_combinations).intersection(set(all_permutations))
    # for permutation in all_permutations:
    #     print("gdict['{}']['pddl'] = gdict['{}']['pddl']".format(task, '_and_'.join(permutation)))
    # print("gdict['{}']['pddl'] = gdict['{}']['pddl']".format(task, list(root_task)[0]))
    # print('\n')

subgoal_to_str_map = {'cool': 'CoolObject', 'heat': 'HeatObject', 'slice': 'SliceObject',
                      'clean': 'CleanObject', 'place': 'PutObjectInReceptacle1'}

# to augment preconditions to the current subgoal based on the previous subgoal
subgoal_precondition_augment = {'cool': ['(isCool ?o)', '(coolable ?o)'], 'heat': ['(isHot ?o)', '(heatable ?o)'],
                                'clean': ['(isClean ?o)', '(cleanable ?o)'],
                                'slice': ['(isSliced ?o)', '(sliceable ?o)']}
subgoal_not_precondition_augment = {'cool': ['(not (isCool ?o))'], 'heat': ['(not (isHot ?o))'],
                                    'clean': ['(not (isClean ?o))'], 'slice': ['(not (isSliced ?o))']}

for task in all_tasks:
    savefile = 'task_' + task + '.pddl'
    with open(savefile, 'w') as outfile:

        if 'and' not in task:
            seq_of_subgoals = task.split('_then_')
            inside_subgoal_definition = False

            for line in all_lines:
                outfile.write(line)

                if subgoal_to_str_map[seq_of_subgoals[0]] in line and seq_of_subgoals[0] != 'place':
                    curr_subgoal = seq_of_subgoals[0]
                    prev_subgoals = None
                    next_subgoals = [x for x in seq_of_subgoals[1:] if x != 'place']
                    inside_subgoal_definition = True

                if subgoal_to_str_map[seq_of_subgoals[1]] in line and seq_of_subgoals[1] != 'place':
                    curr_subgoal = seq_of_subgoals[1]
                    prev_subgoals = [seq_of_subgoals[0]] if seq_of_subgoals[0] != 'place' else None
                    if len(seq_of_subgoals) > 2:
                        next_subgoals = [seq_of_subgoals[2]]
                        if next_subgoals == ['place']:
                            next_subgoals = None
                    else:
                        next_subgoals = None
                    inside_subgoal_definition = True

                if len(seq_of_subgoals) > 2 and subgoal_to_str_map[seq_of_subgoals[2]] in line and \
                        seq_of_subgoals[2] != 'place':
                    curr_subgoal = [seq_of_subgoals[2]]
                    prev_subgoals = [x for x in seq_of_subgoals[:2] if x != 'place']
                    next_subgoals = None
                    inside_subgoal_definition = True

                if inside_subgoal_definition:
                    if 'precondition' in line:
                        if prev_subgoals is not None:
                            for prev_subgoal in prev_subgoals:
                                for augment_condition in subgoal_precondition_augment[prev_subgoal]:
                                    augmented_str = '            ' + augment_condition + '\n'
                                    outfile.write(augmented_str)
                        if next_subgoals is not None:
                            for next_subgoal in next_subgoals:
                                # try:
                                for augment_condition in subgoal_not_precondition_augment[next_subgoal]:
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

                    if second_part != 'place':
                        for prev_subgoal in prev_subgoals:
                            if subgoal_to_str_map[prev_subgoal] in line:
                                inside_subgoal_definition = True
                                next_subgoal = second_part
                        if subgoal_to_str_map[curr_subgoal] in line:
                            inside_subgoal_definition = True
                            next_subgoal = None

                    if inside_subgoal_definition:
                        if 'precondition' in line:
                            if next_subgoal is None:
                                for prev_subgoal in prev_subgoals:
                                    for augment_condition in subgoal_precondition_augment[prev_subgoal]:
                                        augmented_str = '            ' + augment_condition + '\n'
                                        outfile.write(augmented_str)
                            else:
                                for augment_condition in subgoal_not_precondition_augment[next_subgoal]:
                                    augmented_str = '            ' + augment_condition + '\n'
                                    outfile.write(augmented_str)

                        if 'effect' in line:
                            inside_subgoal_definition = False
            else:
                prev_subgoal = first_part
                curr_subgoals = second_part.split('_and_')

                for line in all_lines:
                    outfile.write(line)

                    if subgoal_to_str_map[prev_subgoal] in line:
                        inside_subgoal_definition = True
                        next_subgoals = [x for x in curr_subgoals if x != 'place']
                    for curr_subgoal in curr_subgoals:
                        if subgoal_to_str_map[curr_subgoal] in line and curr_subgoal != 'place':
                            inside_subgoal_definition = True
                            next_subgoals = None

                    if inside_subgoal_definition:
                        if 'precondition' in line:
                            if next_subgoals is None:
                                for augment_condition in subgoal_precondition_augment[prev_subgoal]:
                                    augmented_str = '            ' + augment_condition + '\n'
                                    outfile.write(augmented_str)
                            else:
                                # try:
                                for next_subgoal in next_subgoals:
                                    for augment_condition in subgoal_not_precondition_augment[next_subgoal]:
                                        augmented_str = '            ' + augment_condition + '\n'
                                        outfile.write(augmented_str)
                                # except TypeError:
                                #     print('hi')

                        if 'effect' in line:
                            inside_subgoal_definition = False
