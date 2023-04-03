# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from collections import Counter

GOALS = ['clean_simple',
         'cool_simple',
         'heat_simple',
         'pick_simple',
         'place_simple',
         'slice_simple',
         'clean_and_cool',
         'clean_and_heat',
         'clean_and_place',
         'clean_and_slice',
         'cool_and_place',
         'heat_and_place',
         'slice_and_cool',
         'slice_and_heat',
         'slice_and_place',
         'slice_and_clean_and_place',
         'cool_and_clean_and_place',
         'cool_and_slice_and_place',
         'heat_and_clean_and_place',
         'slice_and_heat_and_place',
         'slice_and_heat_and_clean',
         'cool_and_slice_and_clean',
         'clean_then_cool',
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
         'cool_then_slice_and_place'
         ]

sub_goal_composition = set([x for x in GOALS if len({'cool', 'clean'}.intersection(
    set(x.replace('then','and').split('_and_'))))==2 or len({'heat', 'slice', 'place'}.intersection(
    set(x.replace('then','and').split('_and_'))))==3])

def find_ordering_split(x1, x2, y1, y2):
    ordering_composition = set()
    for x in GOALS:
        if 'then' in x:
            if 'and' in x:
                first, second = x.split('_then_')
                if 'and' in first:
                    # (sub1, sub2), sub3 = first.split('_and_'), second
                    if (x1 in first and second == x2) or (y1 in first and second == y2):
                        ordering_composition.add(x)
                else:
                    # sub1, (sub2, sub3) = first, second.split('_and_')
                    if (first == x1 and x2 in second) or (first == y1 and y2 in second):
                        ordering_composition.add(x)
            else:
                subgoals = x.split('_then_')
                if len(set(subgoals).intersection({x1, x2})) == 2 and \
                        subgoals.index(x1) < subgoals.index(x2):
                    ordering_composition.add(x)
                elif len(set(subgoals).intersection({y1, y2})) == 2 and \
                        subgoals.index(y1) < subgoals.index(y2):
                    ordering_composition.add(x)
    return ordering_composition

# ordering_composition = find_ordering_split(x1='clean', x2='slice', y1='slice', y2='place')
# remaining = set(GOALS) - ordering_composition - sub_goal_composition
remaining = set(GOALS) - sub_goal_composition
m = []
for n in remaining:
    for k in n.replace('then', 'and').split('_and_'):
        if 'simple' in k:
            m.append(k.split('_')[0])
        else:
            m.append(k)
print('train data: {}'.format(Counter(m)))
print(len(remaining))
m = []
for n in sub_goal_composition:
    for k in n.replace('then', 'and').split('_and_'):
        if 'simple' in k:
            m.append(k.split('_')[0])
        else:
            m.append(k)
print('sub-goal-composition: {}'.format(Counter(m)))
print(len(sub_goal_composition))
# print(list(sub_goal_composition))
# print(list(remaining))
context_verb_noun_composition = set([x for x in remaining if len({'heat', 'cool', 'place'}.intersection(
set(x.replace('then', 'and').split('_and_')))) > 0])
# context_verb_noun_composition = context_verb_noun_composition - sub_goal_composition
verb_noun_composition = set([x for x in remaining if len({'heat', 'clean', 'slice', 'place'}.intersection(
set(x.replace('then', 'and').split('_and_')))) > 0])
# verb_noun_composition = verb_noun_composition - sub_goal_composition
for x in verb_noun_composition:
    print("'" + x + "',")
