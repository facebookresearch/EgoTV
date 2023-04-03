# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


#########################################
# Common:
# {obj} - ObjectStr
# {recep} - RecepStr
# usage: .format(obj=constants.OBJECTS[self.object_target], recep=constants.OBJECTS[self.parent_target])

# NOTE: order of and/or conditions matters
#########################################
import constants
from collections import defaultdict

gdict = defaultdict()
for goal in constants.ALL_GOALS:
    gdict[goal] = {'pddl': '', 'templates_pos': []}

###############################################
# LEVEL 1: basic skills
###############################################

####################################################################################
# LEVEL 2: composition of basic skills and interactions with objects + quantifiers
####################################################################################

# pick, clean (in sink), place object
gdict["clean_and_place"] = \
    {
        'pddl':
            '''
        (:goal
            (and
                (forall (?re # receptacle)
                    (not (opened ?re))
                )
                (exists (?r # receptacle)
                    (exists (?o # object)
                        (and 
                            (cleanable ?o)
                            (objectType ?o {obj}Type) 
                            (receptacleType ?r {recep}Type)
                            (isClean ?o)
                            (inReceptacle ?o ?r) 
                        )
                    )
                )
            )
        )
    )
    ''',
        'templates_pos': ['{obj} is cleaned and placed'],
        'templates_neg': ['{obj} is cooled and placed',
                          '{obj} is heated and placed',
                          '{obj} is cleaned and heated']
    }

gdict["place_and_clean"] = gdict["clean_and_place"]

# pick, heat (in microwave/stoveburner), place object
gdict["heat_and_place"] = \
    {
        'pddl':
            '''
        (:goal
            (and
                (exists (?r # receptacle)
                    (exists (?o # object)
                        (and 
                            (heatable ?o)
                            (objectType ?o {obj}Type) 
                            (receptacleType ?r {recep}Type)
                            (isHot ?o)
                            (inReceptacle ?o ?r) 
                        )
                    )
                )
                (forall (?re # receptacle)
                    (not (opened ?re))
                )
            )
        )
    )
    ''',
        'templates_pos': ['hot {obj} is placed',
                          '{obj} is heated and placed'],
        'templates_neg': ['cold {obj} is placed',
                          '{obj} is cleaned and placed',
                          'sliced {obj}']
    }

gdict["place_and_heat"] = gdict["heat_and_place"]

# pick, cool (in refrigerator if not already cool), place object
gdict["cool_and_place"] = \
    {
        'pddl':
            '''
        (:goal
            (and
                (exists (?r # receptacle)
                    (exists (?o # object)
                        (and 
                            (coolable ?o)
                            (objectType ?o {obj}Type) 
                            (receptacleType ?r {recep}Type)
                            (isCool ?o)
                            (inReceptacle ?o ?r) 
                        )
                    )
                )
                (forall (?re # receptacle)
                    (not (opened ?re))
                )
            )
        )
    )
    ''',
        'templates_pos': ['cold {obj} is placed',
                          '{obj} is cooled and placed'],
        'templates_neg': ['hot {obj} is placed',
                          '{obj} is cleaned and placed',
                          '{obj} is cooled and sliced']
    }

gdict["place_and_cool"] = gdict["cool_and_place"]

# slice, place object
gdict["slice_and_place"] = \
    {
        'pddl':
            '''
        (:goal
            (and
                (exists (?r # receptacle)
                    (exists (?o # object)
                        (and 
                            (sliceable ?o)
                            (objectType ?o {obj}Type) 
                            (receptacleType ?r {recep}Type)
                            (isSliced ?o)
                            (inReceptacle ?o ?r) 
                        )
                    )
                )
                (forall (?re # receptacle)
                    (not (opened ?re))
                )
            )
        )
    )
    ''',
        'templates_pos': ['{obj} is sliced and placed',
                          'sliced {obj} is placed'],
        'templates_neg': ['cold {obj} is placed',
                          'clean {obj}',
                          '{obj} is heated and placed']
    }

gdict["place_and_slice"] = gdict["slice_and_place"]

# slice, cool object (in any order)
gdict["slice_and_cool"] = \
    {
        'pddl':
            '''
        (:goal
            (and               
                (exists (?o # object)
                    (exists (?a # agent)
                        (and 
                            (sliceable ?o)
                            (coolable ?o)
                            (objectType ?o {obj}Type) 
                            (isSliced ?o)
                            (isCool ?o)
                            (holds ?a ?o)
                            (holdsAny ?a)
                        )
                    )
                )                
                (forall (?re # receptacle)
                    (not (opened ?re))
                )
            )
        )
    )
    ''',
        'templates_pos': ['cold, sliced {obj}',
                          'sliced, cold {obj}'],
        'templates_neg': ['cold {obj} is placed',
                          'hot, sliced {obj}',
                          '{obj} is sliced and cleaned']
    }

gdict["cool_and_slice"] = gdict["slice_and_cool"]

# slice, heat object (in any order)
gdict["slice_and_heat"] = \
    {
        'pddl':
            '''
        (:goal
            (and               
                (exists (?o # object)
                    (exists (?a # agent)
                        (and 
                            (sliceable ?o)
                            (heatable ?o)
                            (objectType ?o {obj}Type) 
                            (isSliced ?o)
                            (isHot ?o)
                            (holds ?a ?o)
                            (holdsAny ?a)
                        )
                    )
                )                
                (forall (?re # receptacle)
                    (not (opened ?re))
                )
            )
        )
    )
    ''',
        'templates_pos': ['hot, sliced {obj}',
                          'sliced, hot {obj}'],
        'templates_neg': ['cold, sliced {obj}',
                          'sliced {obj} is placed',
                          'hot {obj} is cleaned']
    }

gdict["heat_and_slice"] = gdict["slice_and_heat"]

# slice, clean object (in any order)
gdict["slice_and_clean"] = \
    {
        'pddl':
            '''
        (:goal
            (and               
                (exists (?o # object)
                    (exists (?a # agent)
                        (and 
                            (sliceable ?o)
                            (cleanable ?o)
                            (objectType ?o {obj}Type) 
                            (isSliced ?o)
                            (isClean ?o)
                            (holds ?a ?o)
                            (holdsAny ?a)
                        )
                    )
                )                
                (forall (?re # receptacle)
                    (not (opened ?re))
                )
            )
        )
    )
    ''',
        'templates_pos': ['clean, sliced {obj}',
                          'sliced, clean {obj}'],
        'templates_neg': ['hot, clean {obj}',
                          'sliced, cold {obj}',
                          'hot, clean, sliced {obj}']
    }

gdict["clean_and_slice"] = gdict["slice_and_clean"]

# slice, cool object (in any order)
gdict["clean_and_cool"] = \
    {
        'pddl':
            '''
        (:goal
            (and               
                (exists (?o # object)
                    (exists (?a # agent)
                        (and 
                            (cleanable ?o)
                            (coolable ?o)
                            (objectType ?o {obj}Type) 
                            (isClean ?o)
                            (isCool ?o)
                            (holds ?a ?o)
                            (holdsAny ?a)
                        )
                    )
                )                
                (forall (?re # receptacle)
                    (not (opened ?re))
                )
            )
        )
    )
    ''',
        'templates_pos': ['cold, clean {obj}',
                          'clean, cold {obj}'],
        'templates_neg': ['cold, clean, sliced {obj}',
                          'clean, sliced {obj}',
                          'hot, clean {obj}']
    }

gdict["cool_and_clean"] = gdict["clean_and_cool"]

# slice, heat object (in any order)
gdict["clean_and_heat"] = \
    {
        'pddl':
            '''
        (:goal
            (and               
                (exists (?o # object)
                    (exists (?a # agent)
                        (and 
                            (cleanable ?o)
                            (heatable ?o)
                            (objectType ?o {obj}Type) 
                            (isClean ?o)
                            (isHot ?o)
                            (holds ?a ?o)
                            (holdsAny ?a)
                        )
                    )
                )                
                (forall (?re # receptacle)
                    (not (opened ?re))
                )
            )
        )
    )
    ''',
        'templates_pos': ['hot, clean {obj}',
                          'clean, hot {obj}'],
        'templates_neg': ['cold {obj}',
                          'hot, cleaned, sliced {obj}',
                          'a slice of hot {obj}']
    }

gdict["heat_and_clean"] = gdict["clean_and_heat"]

# pick two instances of an object and place them in a receptacle (e.g: "pick two apples and put them in the sink")
gdict["place_2"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o1 # object)
                                (and 
                                    (objectType ?o1 {obj}Type) 
                                    (receptacleType ?r {recep}Type)
                                    (inReceptacle ?o1 ?r)
                                    (exists (?o2 # object)
                                        (and
                                            (not (= ?o1 ?o2))
                                            (objectType ?o2 {obj}Type)
                                            (receptacleType ?r {recep}Type)
                                            (inReceptacle ?o2 ?r) 
                                        )
                                    )
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['two {obj}s in a {recep}']
    }

# pick and place with a movable receptacle (e.g: "put a apple in a bowl inside the microwave")
gdict["stack_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (and 
                                (receptacleType ?r {recep}Type)
                                (exists (?o # object)
                                    (and
                                        (objectType ?o {obj}Type)
                                        (exists (?mo # object)
                                            (and
                                                (objectType ?mo {mrecep}Type)
                                                (isReceptacleObject ?mo)
                                                (inReceptacleObject ?o ?mo)
                                                (inReceptacle ?mo ?r)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['{obj} in a {mrecep} in a {recep}',
                          '{obj} in a {mrecep} placed in a {recep}',
                          'a {mrecep} containing {obj} in a {recep}',
                          'a {mrecep} containing {obj} placed in a {recep}']
    }

##########################################################################
# LEVEL 3: complex composition of basic skills and interactions with objects
##########################################################################

# pick, heat, place with movable receptacle
gdict["heat_and_stack_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (and 
                                (receptacleType ?r {recep}Type)
                                (exists (?o # object)
                                    (and
                                        (objectType ?o {obj}Type)
                                        (heatable ?o)
                                        (isHot ?o)
                                        (exists (?mo # object)
                                            (and
                                                (objectType ?mo {mrecep}Type)
                                                (isReceptacleObject ?mo)
                                                (inReceptacleObject ?o ?mo)
                                                (inReceptacle ?mo ?r)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['hot {obj} in a {mrecep} in a {recep}',
                          'hot {obj} in a {mrecep} placed in a {recep}',
                          'a {mrecep} containing hot {obj} in a {recep}',
                          'a {mrecep} containing hot {obj} placed in a {recep}']
    }

# pick, cool, place with movable receptacle
gdict["cool_and_stack_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (and 
                                (receptacleType ?r {recep}Type)
                                (exists (?o # object)
                                    (and
                                        (objectType ?o {obj}Type)
                                        (coolable ?o)
                                        (isCool ?o)
                                        (exists (?mo # object)
                                            (and
                                                (objectType ?mo {mrecep}Type)
                                                (isReceptacleObject ?mo)
                                                (inReceptacleObject ?o ?mo)
                                                (inReceptacle ?mo ?r)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['cold {obj} in a {mrecep} in a {recep}',
                          'cold {obj} in a {mrecep} placed in a {recep}',
                          'a {mrecep} containing cold {obj} in a {recep}',
                          'a {mrecep} containing cold {obj} placed in a {recep}']
    }

# pick, clean, place with movable receptacle
gdict["clean_and_stack_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (and 
                                (receptacleType ?r {recep}Type)
                                (exists (?o # object)
                                    (and
                                        (objectType ?o {obj}Type)
                                        (cleanable ?o)
                                        (isClean ?o)
                                        (exists (?mo # object)
                                            (and
                                                (objectType ?mo {mrecep}Type)
                                                (isReceptacleObject ?mo)
                                                (inReceptacleObject ?o ?mo)
                                                (inReceptacle ?mo ?r)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['clean {obj} in a {mrecep} in a {recep}',
                          'clean {obj} in a {mrecep} placed in a {recep}',
                          'a {mrecep} containing clean {obj} in a {recep}',
                          'a {mrecep} containing clean {obj} placed in a {recep}']
    }

# slice, place with movable receptacle
gdict["slice_and_stack_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (and 
                                (receptacleType ?r {recep}Type)
                                (exists (?o # object)
                                    (and
                                        (objectType ?o {obj}Type)
                                        (sliceable ?o)
                                        (isSliced ?o)
                                        (exists (?mo # object)
                                            (and
                                                (objectType ?mo {mrecep}Type)
                                                (isReceptacleObject ?mo)
                                                (inReceptacleObject ?o ?mo)
                                                (inReceptacle ?mo ?r)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['sliced {obj} in a {mrecep} in a {recep}',
                          'sliced {obj} in a {mrecep} placed in a {recep}',
                          'a {mrecep} containing sliced {obj} placed in a {recep}',
                          'a {mrecep} containing sliced {obj} in a {recep}']
    }

# pick, clean, slice, and place object
gdict["slice_and_clean_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                        (exists (?r # receptacle)
                            (exists (?o # object)
                                (and 
                                    (sliceable ?o)
                                    (isSliced ?o)
                                    (cleanable ?o)
                                    (objectType ?o {obj}Type) 
                                    (receptacleType ?r {recep}Type)
                                    (isClean ?o)
                                    (inReceptacle ?o ?r) 
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['clean, sliced {obj} is placed',
                          'sliced, clean {obj} is placed'],
        'templates_neg': ['cold, sliced {obj} is placed',
                          'sliced, hot {obj} is placed',
                          '{obj} is cleaned, sliced and heated',
                          'clean {obj} is sliced and heated',
                          'hot {obj} is sliced and cleaned']

    }

# pick, slice, clean, heat
gdict["slice_and_heat_and_clean"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?o # object)
                            (exists (?a # agent)
                                (and 
                                    (sliceable ?o)
                                    (isSliced ?o)
                                    (heatable ?o)
                                    (cleanable ?o)
                                    (objectType ?o {obj}Type)
                                    (isHot ?o)
                                    (isClean ?o)
                                    (holds ?a ?o)
                                    (holdsAny ?a)
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['hot, clean, sliced {obj}',
                          'sliced, hot, clean {obj}'],
        'templates_neg': ['cold, clean, sliced {obj}',
                          'sliced, cold, clean {obj}',
                          'sliced {obj} is cooled',
                          'clean {obj} is sliced and cooled']
    }

# pick, clean, heat & place
gdict["heat_and_clean_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o # object)
                                (and 
                                    (cleanable ?o)
                                    (isClean ?o)
                                    (heatable ?o)
                                    (objectType ?o {obj}Type) 
                                    (receptacleType ?r {recep}Type)
                                    (isHot ?o)
                                    (inReceptacle ?o ?r) 
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['clean, hot {obj} is placed',
                          'hot, clean {obj} is placed'],
        'templates_neg': ['cold, clean {obj} is placed',
                          'sliced, clean {obj} is placed',
                          'hot, clean {obj} is sliced',
                          'clean {obj} is heated and cooled',
                          'hot {obj} is sliced']
    }

gdict["cool_and_slice_and_clean"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?o # object)
                            (exists (?a # agent)
                                (and 
                                    (sliceable ?o)
                                    (isSliced ?o)
                                    (coolable ?o)
                                    (cleanable ?o)
                                    (objectType ?o {obj}Type)
                                    (isCool ?o)
                                    (isClean ?o)
                                    (holds ?a ?o)
                                    (holdsAny ?a)
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['cold, clean, sliced {obj}',
                          'sliced, cold, clean {obj}'],
        'templates_neg': ['hot, clean, sliced {obj}',
                          'sliced {obj} is cleaned and heated',
                          'sliced, clean, hot {obj}',
                          '{obj} is heated and sliced',
                          '{obj} is heated and cleaned']
    }

# pick, clean, cool & place
gdict["cool_and_clean_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o # object)
                                (and 
                                    (cleanable ?o)
                                    (isClean ?o)
                                    (coolable ?o)
                                    (objectType ?o {obj}Type) 
                                    (receptacleType ?r {recep}Type)
                                    (isCool ?o)
                                    (inReceptacle ?o ?r) 
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['clean, cold {obj} is placed',
                          'cold, clean {obj} is placed'],
        'templates_neg': ['hot, clean {obj} is placed',
                          'hot {obj} is cleaned and cooled',
                          'clean {obj} is cooled and sliced',
                          'sliced {obj} is cleaned and cooled',
                          'clean, sliced {obj} is placed',
                          'cold, clean {obj} is sliced']
    }

# pick, heat, slice & place
gdict["slice_and_heat_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o # object)
                                (and 
                                    (sliceable ?o)
                                    (isSliced ?o)
                                    (heatable ?o)
                                    (objectType ?o {obj}Type) 
                                    (receptacleType ?r {recep}Type)
                                    (isHot ?o)
                                    (inReceptacle ?o ?r) 
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['hot, sliced {obj} is placed',
                          'sliced, hot {obj} is placed'],
        'templates_neg': ['cold, sliced {obj} is cleaned',
                          'hot {obj} is sliced and cleaned',
                          'clean {obj} is picked and heated',
                          'hot, sliced, clean {obj}']
    }

# pick, cool, slice & place
gdict["cool_and_slice_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o # object)
                                (and 
                                    (sliceable ?o)
                                    (isSliced ?o)
                                    (coolable ?o)
                                    (objectType ?o {obj}Type) 
                                    (receptacleType ?r {recep}Type)
                                    (isCool ?o)
                                    (inReceptacle ?o ?r) 
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['cold, sliced {obj} is placed',
                          'sliced, cold {obj} is placed'],
        'templates_neg': ['hot, sliced {obj} is placed',
                          'clean {obj} is cooled and sliced',
                          'cold, clean, sliced {obj}',
                          '{obj} is cleaned and placed',
                          'cold, clean, sliced {obj} is placed']
    }

# pick two instances of sliced object and place them in a receptacle (e.g: "pick two apples and put them in the sink")
gdict["pick_two_obj_and_place_slice"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o1 # object)
                                (and 
                                    (sliceable ?o1)
                                    (isSliced ?o1)
                                    (objectType ?o1 {obj}Type) 
                                    (receptacleType ?r {recep}Type)
                                    (inReceptacle ?o1 ?r)
                                    (exists (?o2 # object)
                                        (and
                                            (not (= ?o1 ?o2))

                                            (sliceable ?o2)
                                            (isSliced ?o2)
                                            (objectType ?o2 {obj}Type)
                                            (receptacleType ?r {recep}Type)
                                            (inReceptacle ?o2 ?r) 
                                        )
                                    )
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['provide two sliced {obj}s in {recep}']
    }

###############################################
# LEVEL 4 long horizon tasks
###############################################

# toggle the state of a toggleable object (e.g: "toggle the lightswitch") while holding another, sliced one.
gdict["look_at_obj_in_light_slice"] = \
    {
        'pddl':
            '''
                (:goal
                     (and
                         (exists (?ot # object 
                                  ?a # agent 
                                  ?l # location)
                             (and
                                 (sliceable ?o)
                                (isSliced ?o)
                                 (objectType ?ot {toggle}Type)
                                 (toggleable ?ot)
                                 (isToggled ?ot)
                                 (objectAtLocation ?ot ?l)
                                 (atLocation ?a ?l)
                             )
                         )
                         (exists (?o # object
                                  ?a # agent)
                             (and 
                                 (objectType ?o {obj}Type)
                                 (holds ?a ?o)
                             )
                         )
                     )
                )
            )
            ''',
        'templates_pos': ['look at sliced {obj} under the {toggle}',
                          'examine the sliced {obj} with the {toggle}']
    }

# put all objects of a type inside in one receptacle (e.g: "put all the mugs in the microwave")
gdict["place_all_obj_type_into_recep"] = \
    {
        'pddl':
            '''
        (:goal
            (and
                (forall (?re # receptacle)
                    (not (opened ?re))
                )
                (exists (?r # receptacle)
                    (forall (?o # object)
                        (or
                            (and
                                (objectType ?o {obj}Type)
                                (receptacleType ?r {recep}Type)
                                (inReceptacle ?o ?r)
                            )
                            (or
                                (not (objectType ?o {obj}Type))
                            )
                        )
                    )
                )
            )
        )
    )
    ''',
        'templates_pos': ['put all {obj}s in {recep}',
                          'find all {obj}s and put them in {recep}']
    }

# pick three instances of an object and place them in a receptacle (e.g: "pick three apples and put them in the sink")
# NOTE: doesn't work
gdict["pick_three_obj_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o1 # object)
                                (and 
                                    (objectType ?o1 {obj}Type) 
                                    (receptacleType ?r {recep}Type)
                                    (inReceptacle ?o1 ?r) 

                                    (exists (?o2 # object)
                                        (and
                                            (not (= ?o1 ?o2))
                                            (objectType ?o2 {obj}Type)
                                            ;(receptacleType ?r {recep}Type)
                                            (inReceptacle ?o2 ?r) 
                                            (exists (?o3 # object)
                                                (and
                                                    (not (= ?o1 ?o3))
                                                    (not (= ?o2 ?o3))
                                                    (objectType ?o3 {obj}Type)
                                                    ;(receptacleType ?r {recep}Type)
                                                    (inReceptacle ?o3 ?r)
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['put three {obj}s in {recep}',
                          'find three {obj}s and put them in {recep}']
    }

gdict["pick_heat_and_place_with_movable_recep"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (and 
                                (receptacleType ?r {recep}Type)
                                (exists (?o # object)
                                    (and
                                        (objectType ?o {obj}Type)
                                        (heatable ?o)
                                        (isHot ?o)
                                        (exists (?mo # object)
                                            (and
                                                (objectType ?mo {mrecep}Type)
                                                (isReceptacleObject ?mo)
                                                (inReceptacleObject ?o ?mo)
                                                (inReceptacle ?mo ?r)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['Provide hot {mrecep} of {obj} in {recep}']
    }

gdict["pick_cool_and_place_with_movable_recep"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (and 
                                (receptacleType ?r {recep}Type)
                                (exists (?o # object)
                                    (and
                                        (objectType ?o {obj}Type)
                                        (coolable ?o)
                                        (isCool ?o)
                                        (exists (?mo # object)
                                            (and
                                                (objectType ?mo {mrecep}Type)
                                                (isReceptacleObject ?mo)
                                                (inReceptacleObject ?o ?mo)
                                                (inReceptacle ?mo ?r)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['Provide cold {mrecep} of {obj} in {recep}']
    }

gdict["pick_clean_and_place_with_movable_recep"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (and 
                                (receptacleType ?r {recep}Type)
                                (exists (?o # object)
                                    (and
                                        (objectType ?o {obj}Type)
                                        (cleanable ?o)
                                        (isClean ?o)
                                        (exists (?mo # object)
                                            (and
                                                (objectType ?mo {mrecep}Type)
                                                (isReceptacleObject ?mo)
                                                (inReceptacleObject ?o ?mo)
                                                (inReceptacle ?mo ?r)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['Provide cold {mrecep} of {obj} in {recep}']
    }

###########################################################################################################
###########################################################################################################
# Axis 2: partial ordering
# Level 2
###########################################################################################################
############################################################################################################


###############################################
# Axis 2: partial ordering
# Level 3
###############################################

# pick, clean, then slice, then place object
gdict["clean_then_slice_then_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                        (exists (?r # receptacle)
                            (exists (?o # object)
                                (and 
                                    (sliceable ?o)
                                    (isSliced ?o)
                                    (cleanable ?o)
                                    (objectType ?o {obj}Type) 
                                    (receptacleType ?r {recep}Type)
                                    (isClean ?o)
                                    (inReceptacle ?o ?r) 
                                )
                            )
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['clean {obj} is sliced before placing',
                          'clean {obj} is placed after slicing'],
        'templates_neg': ['{obj} is sliced, then cleaned, then placed',
                          '{obj} is sliced before cleaning and placing',
                          '{obj} is cleaned after slicing and placing']

    }

# pick, heat, slice & place
gdict["heat_then_slice_then_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o # object)
                                (and 
                                    (sliceable ?o)
                                    (isSliced ?o)
                                    (heatable ?o)
                                    (objectType ?o {obj}Type) 
                                    (receptacleType ?r {recep}Type)
                                    (isHot ?o)
                                    (inReceptacle ?o ?r) 
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['hot {obj} is sliced before placing',
                          'hot {obj} is placed after slicing'],
        'template_neg': ['{obj} is sliced, then heated, then placed',
                         '{obj} is sliced before heating and placing',
                         '{obj} is picked and heated after slicing']
    }

# pick, cool, slice & place
gdict["cool_then_slice_then_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o # object)
                                (and 
                                    (sliceable ?o)
                                    (isSliced ?o)
                                    (coolable ?o)
                                    (objectType ?o {obj}Type) 
                                    (receptacleType ?r {recep}Type)
                                    (isCool ?o)
                                    (inReceptacle ?o ?r) 
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['cold {obj} is sliced before placing',
                          'cold {obj} is placed after slicing'],
        'template_neg': ['{obj} is sliced, then cooled, then placed',
                         '{obj} is sliced before cooling and placing',
                         '{obj} is cooled after slicing']
    }

# pick two objects, slice them, then place them in a receptacle
gdict["pick_two_obj_then_slice_and_place"] = \
    {
        'pddl':
            '''
                (:goal
                    (and
                        (exists (?r # receptacle)
                            (exists (?o1 # object)
                                (and 
                                    (sliceable ?o1)
                                    (isSliced ?o1)
                                    (objectType ?o1 {obj}Type) 
                                    (receptacleType ?r {recep}Type)
                                    (inReceptacle ?o1 ?r)
                                    (exists (?o2 # object)
                                        (and
                                            (not (= ?o1 ?o2))
                                            (sliceable ?o2)
                                            (isSliced ?o2)
                                            (objectType ?o2 {obj}Type)
                                            (receptacleType ?r {recep}Type)
                                            (inReceptacle ?o2 ?r) 
                                        )
                                    )
                                )
                            )
                        )
                        (forall (?re # receptacle)
                            (not (opened ?re))
                        )
                    )
                )
            )
            ''',
        'templates_pos': ['slice two {obj}s and put them in {recep}',
                          'put two sliced {obj}s in {recep}']
    }

#####################################################################

gdict['clean_then_cool']['pddl'] = gdict['cool_and_clean']['pddl']
gdict['clean_then_cool']['templates_pos'] = ['{obj} is cleaned and then cooled',
                                             '{obj} is cleaned before cooling',
                                             '{obj} is cooled after cleaning']
gdict['clean_then_cool']['templates_neg'] = ['clean {obj} is heated',
                                             'hot {obj} is cooled',
                                             'cold {obj} is cleaned',
                                             'clean, cold {obj} is sliced',
                                             '{obj} is cooled, then cleaned',
                                             '{obj} is cleaned after cooling',
                                             '{obj} is cooled after cleaning']


gdict['cool_then_clean']['pddl'] = gdict['cool_and_clean']['pddl']
gdict['cool_then_clean']['templates_pos'] = ['{obj} is cooled and then cleaned',
                                             '{obj} is cooled before cleaning',
                                             '{obj} is cleaned after cooling']
gdict['cool_then_clean']['templates_neg'] = ['hot {obj} is cleaned',
                                             'clean {obj} is cooled',
                                             'cold, clean {obj} is sliced',
                                             '{obj} is cooled, then sliced',
                                             '{obj} is cleaned, then cooled',
                                             '{obj} is cleaned before cooling',
                                             '{obj} is cooled after cleaning']


gdict['clean_then_heat']['pddl'] = gdict['clean_and_heat']['pddl']
gdict['clean_then_heat']['templates_pos'] = ['{obj} is cleaned and then heated',
                                             '{obj} is cleaned before heating',
                                             '{obj} is heated after cleaning']
gdict['clean_then_heat']['templates_neg'] = ['clean {obj} is cooled',
                                             'hot {obj} is cleaned',
                                             'sliced {obj} is heated',
                                             'clean, hot {obj} is sliced',
                                             '{obj} is heated, then cleaned',
                                             '{obj} is cleaned after heating',
                                             '{obj} is heated before cleaning']

gdict['heat_then_clean']['pddl'] = gdict['clean_and_heat']['pddl']
gdict['heat_then_clean']['templates_pos'] = ['{obj} is heated and then cleaned',
                                             '{obj} is heated before cleaning',
                                             '{obj} is cleaned after heating']
gdict['heat_then_clean']['templates_neg'] = ['clean {obj} is heated',
                                             'hot {obj} is placed and sliced',
                                             'cold {obj} is cleaned',
                                             'hot, clean {obj} is sliced',
                                             '{obj} is cleaned, then heated',
                                             '{obj} is cleaned before heating',
                                             '{obj} is heated after cleaning']

gdict['cool_then_slice']['pddl'] = gdict['cool_and_slice']['pddl']
gdict['cool_then_slice']['templates_pos'] = ['{obj} is cooled and then sliced',
                                             '{obj} is cooled before slicing',
                                             '{obj} is sliced after cooling']
gdict['cool_then_slice']['templates_neg'] = ['cold {obj} is cleaned',
                                             'hot {obj} is sliced',
                                             'cold, sliced {obj} is cleaned',
                                             'sliced {obj} is cooled',
                                             '{obj} is sliced, then cooled',
                                             '{obj} is sliced before cooling',
                                             '{obj} is cooled after slicing']

gdict['slice_then_cool']['pddl'] = gdict['cool_and_slice']['pddl']
gdict['slice_then_cool']['templates_pos'] = ['{obj} is sliced and then cooled',
                                             '{obj} is sliced before cooling',
                                             '{obj} is cooled after slicing']
gdict['slice_then_cool']['templates_neg'] = ['cold {obj} is sliced',
                                             'sliced {obj} is heated',
                                             'cold, sliced {obj} is cleaned',
                                             '{obj} is cooled, then sliced',
                                             '{obj} is cooled before slicing',
                                             '{obj} is sliced after cooling']

gdict['clean_then_slice']['pddl'] = gdict['clean_and_slice']['pddl']
gdict['clean_then_slice']['templates_pos'] = ['{obj} is cleaned and then sliced',
                                              '{obj} is cleaned before slicing',
                                              '{obj} is sliced after cleaning']
gdict['clean_then_slice']['templates_neg'] = ['cold {obj} is sliced',
                                              'sliced {obj} is cleaned',
                                              'clean, sliced {obj} is cooled',
                                              '{obj} is sliced, then cleaned',
                                              '{obj} is sliced before cleaning',
                                              '{obj} is cleaned after slicing']

gdict['slice_then_clean']['pddl'] = gdict['clean_and_slice']['pddl']
gdict['slice_then_clean']['templates_pos'] = ['{obj} is sliced and then cleaned',
                                              '{obj} is sliced before cleaning',
                                              '{obj} is cleaned after slicing']
gdict['slice_then_clean']['templates_neg'] = ['clean {obj} is sliced',
                                              'sliced {obj} is cooled',
                                              'clean, sliced {obj} is cooled',
                                              '{obj} is cleaned, then sliced',
                                              '{obj} is cleaned before slicing',
                                              '{obj} is sliced after cleaning']

gdict['heat_then_slice']['pddl'] = gdict['heat_and_slice']['pddl']
gdict['heat_then_slice']['templates_pos'] = ['{obj} is heated and then sliced',
                                             '{obj} is heated before slicing',
                                             '{obj} is sliced after heating']
gdict['heat_then_slice']['templates_neg'] = ['sliced {obj} is heated',
                                             'cold {obj} is sliced',
                                             'hot {obj} is cleaned',
                                             '{obj} is sliced, then heated',
                                             '{obj} is sliced before heating',
                                             '{obj} is heated after slicing']
gdict['heat_then_slice']['templates_neg'] = ['sliced {obj} is heated',
                                             'cold {obj} is sliced',
                                             'hot {obj} is cleaned',
                                             '{obj} is sliced, then heated',
                                             '{obj} is sliced before heating',
                                             '{obj} is heated after slicing']

gdict['slice_then_heat']['pddl'] = gdict['heat_and_slice']['pddl']
gdict['slice_then_heat']['templates_pos'] = ['{obj} is sliced and then heated',
                                             '{obj} is sliced before heating',
                                             '{obj} is heated after slicing']
gdict['slice_then_heat']['templates_neg'] = ['sliced {obj} is cooled',
                                             'hot {obj} is sliced',
                                             'hot, sliced {obj} is cleaned',
                                             '{obj} is heated, then sliced',
                                             '{obj} is heated before slicing',
                                             '{obj} is sliced after heating']

gdict['slice_then_place']['pddl'] = gdict['slice_and_place']['pddl']
gdict['slice_then_place']['templates_pos'] = ['{obj} is sliced and then placed',
                                              '{obj} is sliced before placing',
                                              '{obj} is placed after slicing']
gdict['slice_then_place']['templates_neg'] = ['clean {obj} is placed',
                                              'sliced {obj} is cooled and placed',
                                              '{obj} is heated, then placed',
                                              '{obj} is cooled before placing',
                                              '{obj} is placed after cleaning']

gdict['clean_then_place']['pddl'] = gdict['clean_and_place']['pddl']
gdict['clean_then_place']['templates_pos'] = ['{obj} is cleaned and then placed',
                                              '{obj} is cleaned before placing',
                                              '{obj} is placed after cleaning']
gdict['clean_then_place']['templates_neg'] = ['cold {obj} is placed',
                                              'clean {obj} is sliced and placed',
                                              '{obj} is heated, then placed',
                                              '{obj} is cooled before placing',
                                              '{obj} is placed after slicing']

gdict['cool_then_place']['pddl'] = gdict['cool_and_place']['pddl']
gdict['cool_then_place']['templates_pos'] = ['{obj} is cooled and then placed',
                                             '{obj} is cooled before placing',
                                             '{obj} is placed after cooling']
gdict['cool_then_place']['templates_neg'] = ['hot {obj} is place',
                                             'cold {obj} is sliced and placed',
                                             '{obj} is heated, then placed',
                                             '{obj} is cleaned before placing',
                                             '{obj} is placed after slicing']

gdict['heat_then_place']['pddl'] = gdict['heat_and_place']['pddl']
gdict['heat_then_place']['templates_pos'] = ['{obj} is heated and then placed',
                                             '{obj} is heated before placing',
                                             '{obj} is placed after heating']
gdict['heat_then_place']['templates_neg'] = ['cold {obj} is placed',
                                             'hot {obj} is cleaned and placed',
                                             '{obj} is cooled, then placed',
                                             '{obj} is cleaned before placing',
                                             '{obj} is placed after slicing']

# ['cool_and_slice_and_place', 'cool_and_clean_and_place', 'cool_and_slice_and_clean', 'slice_and_heat_and_clean',
# 'slice_and_heat_and_place', 'slice_and_clean_and_place', 'heat_and_clean_and_place']
###########################################################################################
###########################################################################################

gdict['clean_then_cool_then_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['clean_then_cool_then_place']['templates_pos'] = ['{obj} is cleaned, then cooled, and then placed']
gdict['clean_then_cool_then_place']['templates_neg'] = ['{obj} is cooled, then cleaned, then placed',
                                                        '{obj} is cleaned after cooling',
                                                        '{obj} is cooled before cleaning and placing']

gdict['clean_then_cool_then_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_then_cool_then_slice']['templates_pos'] = ['{obj} is cleaned, then cooled, and then sliced']
gdict['clean_then_cool_then_slice']['templates_neg'] = ['{obj} is cleaned, then sliced, then cooled',
                                                        '{obj} is cooled, then cleaned, then sliced',
                                                        '{obj} is sliced, then cleaned, then cooled',
                                                        '{obj} is cooled and sliced, then cleaned',
                                                        '{obj} is cleaned after cooling and slicing',
                                                        '{obj} is cooled and sliced before cleaning']

gdict['clean_then_heat_then_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['clean_then_heat_then_place']['templates_pos'] = ['{obj} is cleaned, then heated, and then placed']
gdict['clean_then_heat_then_place']['templates_neg'] = ['{obj} is heated, then cleaned then placed',
                                                        '{obj} is heated, then cleaned, then placed',
                                                        '{obj} is cleaned after heating',
                                                        '{obj} is heated before cleaning and placing']

gdict['clean_then_heat_then_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_then_heat_then_slice']['templates_pos'] = ['{obj} is cleaned, then heated, and then sliced']
gdict['clean_then_heat_then_slice']['templates_neg'] = ['{obj} is heated, then cleaned, then sliced',
                                                        '{obj} is sliced, then heated, then cleaned',
                                                        '{obj} is heated, then sliced, then cleaned',
                                                        '{obj} is sliced, then cleaned, then heated',
                                                        '{obj} is heated after cleaning and slicing',
                                                        '{obj} cleaned and sliced before heating',
                                                        '{obj} is cleaned and heated after slicing',
                                                        '{obj} is sliced before cleaning and heating']

gdict['clean_then_slice_then_cool']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_then_slice_then_cool']['templates_pos'] = ['{obj} is cleaned, then sliced, and then cooled']
gdict['clean_then_slice_then_cool']['templates_neg'] = ['{obj} is cooled, then cleaned, then sliced',
                                                        '{obj} is cleaned, and cooled, then sliced',
                                                        '{obj} is sliced after cleaning and cooling',
                                                        '{obj} is cleaned and cooled before slicing',
                                                        '{obj} is sliced, then cleaned, then cooled',
                                                        '{obj} is cleaned after cooling and slicing',
                                                        '{obj} is cooled and sliced before cleaning']

gdict['clean_then_slice_then_heat']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_then_slice_then_heat']['templates_pos'] = ['{obj} is cleaned, then sliced, and then heated']
gdict['clean_then_slice_then_heat']['templates_neg'] = ['{obj} is cleaned, then heated, then sliced',
                                                        '{obj} is heated, then cleaned, then sliced',
                                                        '{obj} is sliced, then heated, then cleaned',
                                                        '{obj} is heated, then sliced, then cleaned',
                                                        '{obj} is sliced, then cleaned, then heated',
                                                        '{obj} is cleaned and heated after slicing',
                                                        '{obj} is sliced before cleaning and heating',
                                                        '{obj} is sliced and cleaned after heating',
                                                        '{obj} is heated before cleaning and slicing']

gdict['cool_then_clean_then_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['cool_then_clean_then_place']['templates_pos'] = ['{obj} is cooled, then cleaned, and then placed']
gdict['cool_then_clean_then_place']['templates_neg'] = ['{obj} is cleaned, then cooled, then placed',
                                                        '{obj} is cooled after cleaning',
                                                        '{obj} is cleaned before cooling and placing']

gdict['cool_then_clean_then_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['cool_then_clean_then_slice']['templates_pos'] = ['{obj} is cooled, then cleaned, and then sliced']
gdict['cool_then_clean_then_slice']['templates_neg'] = ['{obj} is cleaned, then cooled, then sliced',
                                                        '{obj} is cleaned, then sliced, then cooled',
                                                        '{obj} is sliced, then cleaned, then cooled',
                                                        '{obj} is cleaned, and cooled, then sliced',
                                                        '{obj} is cleaned after cooling and slicing',
                                                        '{obj} is cooled and sliced before cleaning',
                                                        '{obj} is cooled after cleaning and slicing',
                                                        '{obj} is cleaned and sliced before cooling']

gdict['cool_then_slice_then_clean']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['cool_then_slice_then_clean']['templates_pos'] = ['{obj} is cooled, then sliced, and then cleaned']
gdict['cool_then_slice_then_clean']['templates_neg'] = ['{obj} is cooled, then cleaned then sliced',
                                                        '{obj} is cleaned, then cooled then sliced',
                                                        '{obj} is cleaned, then sliced, then cooled',
                                                        '{obj} is sliced, then cleaned then cooled',
                                                        '{obj} is cooled after cleaning and slicing',
                                                        '{obj} is cleaned and sliced before cooling',
                                                        '{obj} is sliced after cooling and cleaning',
                                                        '{obj} is cooled and cleaned before slicing']

gdict['heat_then_clean_then_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['heat_then_clean_then_place']['templates_pos'] = ['{obj} is heated, then cleaned, and then placed']
gdict['heat_then_clean_then_place']['templates_neg'] = ['{obj} is cleaned, then heated, then placed',
                                                        '{obj} is cleaned, then heated, then placed',
                                                        '{obj} is heated after cleaning',
                                                        '{obj} is cleaned before heating and placing']

gdict['heat_then_clean_then_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['heat_then_clean_then_slice']['templates_pos'] = ['{obj} is heated, then cleaned, and then sliced']
gdict['heat_then_clean_then_slice']['templates_neg'] = ['{obj} is cleaned, then heated, then sliced',
                                                        '{obj} is cleaned, then heated, then sliced',
                                                        '{obj} is sliced, then heated, then cleaned',
                                                        '{obj} is heated, then sliced, then cleaned',
                                                        '{obj} heated, then sliced, then cleaned',
                                                        '{obj} is sliced, then cleaned then heated',
                                                        '{obj} is heated after cleaning and slicing',
                                                        '{obj} cleaned and sliced before heating',
                                                        '{obj} is cleaned and heated after slicing',
                                                        '{obj} is sliced before cleaning and heating']

gdict['heat_then_slice_then_clean']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['heat_then_slice_then_clean']['templates_pos'] = ['{obj} is heated, then sliced, and then cleaned']
gdict['heat_then_slice_then_clean']['templates_neg'] = ['{obj} is cleaned, then heated, then sliced',
                                                        '{obj} cleaned, then heated,then sliced',
                                                        '{obj} is sliced, then heated, then cleaned',
                                                        '{obj} is sliced, then cleaned, then heated',
                                                        '{obj} is heated after cleaning and slicing',
                                                        '{obj} cleaned and sliced before heating',
                                                        '{obj} is cleaned before heating and slicing',
                                                        '{obj} is heated and sliced after cleaning']

gdict['slice_then_clean_then_cool']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['slice_then_clean_then_cool']['templates_pos'] = ['{obj} is sliced, then cleaned, and then cooled']
gdict['slice_then_clean_then_cool']['templates_neg'] = ['{obj} is cooled, then cleaned, then sliced',
                                                        '{obj} is cleaned, then cooled, then sliced',
                                                        '{obj} is cleaned, then sliced, then cooled',
                                                        '{obj} is sliced, then cooled, then cleaned',
                                                        '{obj} is sliced after cooling and cleaning',
                                                        '{obj} is cleaned and cooled before slicing',
                                                        '{obj} is cooled before cleaning and slicing',
                                                        '{obj} is cleaned and sliced after cooling']

gdict['slice_then_clean_then_heat']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['slice_then_clean_then_heat']['templates_pos'] = ['{obj} is sliced, then cleaned, and then heated']
gdict['slice_then_clean_then_heat']['templates_neg'] = ['{obj} is heated, then cleaned, then sliced',
                                                        '{obj} is cleaned, then heated, then sliced',
                                                        '{obj} is cleaned, then sliced, then heated',
                                                        '{obj} is sliced, then heated, then cleaned',
                                                        '{obj} is sliced after cleaning and heating',
                                                        '{obj} is cleaned and heated before slicing',
                                                        '{obj} is heated before cleaning and slicing',
                                                        '{obj} is cleaned and sliced after heating']

gdict['slice_then_clean_then_place']['pddl'] = gdict['slice_and_clean_and_place']['pddl']
gdict['slice_then_clean_then_place']['templates_pos'] = ['{obj} is sliced, then cleaned, and then placed']
gdict['slice_then_clean_then_place']['templates_neg'] = ['{obj} is cleaned, then sliced, then placed',
                                                         '{obj} is sliced after cleaning',
                                                         '{obj} is cleaned before slicing and placing']

gdict['slice_then_cool_then_clean']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['slice_then_cool_then_clean']['templates_pos'] = ['{obj} is sliced, then cooled, and then cleaned']
gdict['slice_then_cool_then_clean']['templates_neg'] = ['{obj} is cooled, then cleaned, then sliced',
                                                        '{obj} is cleaned, then cooled, then sliced',
                                                        '{obj} is cleaned, then sliced, then cooled',
                                                        '{obj} is sliced, then cleaned, then cooled',
                                                        '{obj} is sliced after cooling and cleaning',
                                                        '{obj} is cleaned and cooled before slicing',
                                                        '{obj} is cleaned before cooling and slicing',
                                                        '{obj} is cooled and sliced after cleaning']

gdict['slice_then_cool_then_place']['pddl'] = gdict['cool_and_slice_and_place']['pddl']
gdict['slice_then_cool_then_place']['templates_pos'] = ['{obj} is sliced, then cooled, and then placed']
gdict['slice_then_cool_then_place']['templates_neg'] = ['{obj} is cooled, then sliced, then placed',
                                                        '{obj} is sliced after cooling',
                                                        '{obj} is cooled before slicing and placing']

gdict['slice_then_heat_then_clean']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['slice_then_heat_then_clean']['templates_pos'] = ['{obj} is sliced, then heated, and then cleaned']
gdict['slice_then_heat_then_clean']['templates_neg'] = ['{obj} is heated, then cleaned then sliced',
                                                        '{obj} is cleaned, then heated, then sliced',
                                                        '{obj} is cleaned, then sliced, then heated',
                                                        '{obj} is sliced, then cleaned, then heated',
                                                        '{obj} is sliced after heating and cleaning',
                                                        '{obj} is cleaned and heated before slicing',
                                                        '{obj} is cleaned before heating and slicing',
                                                        '{obj} is heated and sliced after cleaning']

gdict['slice_then_heat_then_place']['pddl'] = gdict['slice_and_heat_and_place']['pddl']
gdict['slice_then_heat_then_place']['templates_pos'] = ['{obj} is sliced, then heated, and then placed']
gdict['slice_then_heat_then_place']['templates_neg'] = ['{obj} is heated, then sliced, then placed',
                                                        '{obj} is sliced after heating',
                                                        '{obj} is heated before slicing and placing']

###########################################################################################
###########################################################################################

gdict['clean_and_cool_then_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['clean_and_cool_then_place']['templates_pos'] = ['{obj} is cleaned and cooled, and then placed',
                                                       '{obj} is placed after cleaning and cooling',
                                                       '{obj} is cleaned and cooled before placing']
gdict['clean_and_cool_then_place']['templates_neg'] = ['{obj} is cleaned and heated before placing',
                                                       '{obj} is cooled and heated before placing',
                                                       '{obj} is cleaned and cooled before slicing',
                                                       'clean, sliced {obj} is placed',
                                                       'hot, clean {obj} is placed',
                                                       'cold, clean {obj} is sliced',
                                                       '{obj} is placed after cleaning and heating',
                                                       '{obj} is placed after slicing and heating']

gdict['clean_and_cool_then_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_and_cool_then_slice']['templates_pos'] = ['{obj} is cleaned and cooled, and then sliced',
                                                       '{obj} is sliced after cleaning and cooling',
                                                       '{obj} is cleaned and cooled before sliced']
gdict['clean_and_cool_then_slice']['templates_neg'] = ['{obj} is sliced before cleaning and cooling',
                                                       '{obj} is cleaned and cooled after slicing',
                                                       '{obj} is cleaned and heated before slicing',
                                                       '{obj} is sliced before cleaning and heating',
                                                       '{obj} is cooled and heated before slicing',
                                                       '{obj} is sliced before cooling and heating',
                                                       '{obj} is cleaned, cooled, and then heated']

gdict['clean_and_heat_then_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['clean_and_heat_then_place']['templates_pos'] = ['{obj} is cleaned and heated, and then placed',
                                                       '{obj} is placed after cleaning and heating',
                                                       '{obj} is cleaned and heated before placing']
gdict['clean_and_heat_then_place']['templates_neg'] = ['{obj} is cleaned and cooled before placing',
                                                       '{obj} is cooled and cleaned before placing',
                                                       '{obj} is cleaned and heated before slicing',
                                                       'clean, sliced {obj} is placed',
                                                       'cold, clean {obj} is placed',
                                                       'hot, clean {obj} is sliced',
                                                       '{obj} is placed after cleaning and cooling',
                                                       '{obj} is placed after slicing and heating']

gdict['clean_and_heat_then_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_and_heat_then_slice']['templates_pos'] = ['{obj} is cleaned and heated, and then sliced',
                                                       '{obj} is sliced after cleaning and heating',
                                                       '{obj} is cleaned and heated before slicing']
gdict['clean_and_heat_then_slice']['templates_neg'] = ['{obj} is sliced before cleaning and heating',
                                                       '{obj} is cleaned after heating and slicing',
                                                       '{obj} is heated and cleaned after slicing',
                                                       '{obj} is heated and sliced before cleaning',
                                                       '{obj} is cleaned and cooled then sliced',
                                                       '{obj} is sliced, then cleaned and heated']

gdict['clean_and_slice_then_cool']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_and_slice_then_cool']['templates_pos'] = ['{obj} is cleaned and sliced, and then cooled',
                                                       '{obj} is cooled after cleaning and slicing',
                                                       '{obj} is cleaned and sliced before cooling']
gdict['clean_and_slice_then_cool']['templates_neg'] = ['{obj} is cooled, then cleaned and sliced',
                                                       '{obj} is cooled before cleaning and slicing',
                                                       '{obj} is cleaned after cooling and slicing',
                                                       '{obj} is cooled and sliced before cleaning',
                                                       '{obj} is cleaned and sliced after cooling',
                                                       '{obj} is cleaned and sliced, then heated']

gdict['clean_and_slice_then_heat']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_and_slice_then_heat']['templates_pos'] = ['{obj} is cleaned and sliced, and then heated',
                                                       '{obj} is heated after cleaning and slicing',
                                                       '{obj} is cleaned and sliced before heating']
gdict['clean_and_slice_then_heat']['templates_neg'] = ['{obj} is heated, then cleaned and sliced',
                                                       '{obj} is heated before cleaning and slicing',
                                                       '{obj} is cleaned after heating and slicing',
                                                       '{obj} is heated and sliced before cleaning',
                                                       '{obj} is cleaned and sliced',
                                                       '{obj} is cleaned and sliced, then cooled']

# cool_and_clean_then_place, cool_and_clean_then_slice, heat_and_clean_then_place,
# heat_and_clean_then_slice, slice_and_clean_then_cool, slice_and_clean_then_heat, slice_and_cool_then_clean
# clean_then_slice_and_cool, clean_then_slice_and_heat, cool_then_slice_and_clean, heat_then_slice_and_clean
# slice_then_cool_and_clean, slice_then_heat_and_clean, slice_and_heat_then_clean
gdict['cool_and_slice_then_clean']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['cool_and_slice_then_clean']['templates_pos'] = ['{obj} is cooled and sliced, and then cleaned',
                                                       '{obj} is cleaned after cooling and slicing',
                                                       '{obj} is cooled and sliced before cleaning']
gdict['cool_and_slice_then_clean']['templates_neg'] = ['{obj} is cleaned before cooling and slicing',
                                                       '{obj} is cleaned and sliced before cooling',
                                                       '{obj} is cooled after cleaning and slicing',
                                                       '{obj} is cooled and sliced after cleaning',
                                                       '{obj} is heated and sliced, then cleaned',
                                                       '{obj} is cooled and cleaned before slicing']

gdict['heat_and_slice_then_clean']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['heat_and_slice_then_clean']['templates_pos'] = ['{obj} is heated and sliced, and then cleaned',
                                                       '{obj} is cleaned after heating and slicing',
                                                       '{obj} is heated and sliced before cleaning']
gdict['heat_and_slice_then_clean']['templates_neg'] = ['{obj} is cleaned before heating and slicing',
                                                       '{obj} is cleaned and sliced before heating',
                                                       '{obj} is heated after cleaning and slicing',
                                                       '{obj} is heated and sliced after cleaning',
                                                       '{obj} is cooled and sliced, then cleaned',
                                                       '{obj} is heated and cleaned before slicing']

gdict['slice_and_clean_then_place']['pddl'] = gdict['slice_and_clean_and_place']['pddl']
gdict['slice_and_clean_then_place']['templates_pos'] = ['{obj} is cleaned and sliced, and then placed',
                                                        '{obj} is placed after cleaning and slicing',
                                                        '{obj} is cleaned and sliced before placing']
gdict['slice_and_clean_then_place']['templates_neg'] = ['{obj} is cleaned and cooled before placing',
                                                        '{obj} is cooled and cleaned before placing',
                                                        '{obj} is cleaned and heated before slicing',
                                                        'hot, sliced {obj} is placed',
                                                        'cold, clean {obj} is placed',
                                                        'hot, clean {obj} is sliced',
                                                        '{obj} is placed after cleaning and cooling',
                                                        '{obj} is placed after slicing and heating']

gdict['slice_and_cool_then_place']['pddl'] = gdict['cool_and_slice_and_place']['pddl']
gdict['slice_and_cool_then_place']['templates_pos'] = ['{obj} is cooled and sliced, and then placed',
                                                        '{obj} is placed after cooling and slicing',
                                                        '{obj} is cooled and sliced before placing']
gdict['slice_and_cool_then_place']['templates_neg'] = ['{obj} is cleaned and cooled before placing',
                                                       '{obj} is heated and sliced before placing',
                                                       '{obj} is cleaned and cooled before slicing',
                                                       'hot, sliced {obj} is placed',
                                                       'cold, clean {obj} is placed',
                                                       'cold, clean {obj} is sliced',
                                                       '{obj} is placed after cleaning and cooling',
                                                       '{obj} is placed after slicing and heating']

gdict['slice_and_heat_then_place']['pddl'] = gdict['slice_and_heat_and_place']['pddl']
gdict['slice_and_heat_then_place']['templates_pos'] = ['{obj} is sliced and heated, and then placed',
                                                       '{obj} is placed after slicing and heating',
                                                       '{obj} is sliced and heated before placing']
gdict['slice_and_heat_then_place']['templates_neg'] = ['{obj} is cleaned and heated before placing',
                                                       '{obj} is cooled and sliced before placing',
                                                       '{obj} is cleaned and heated before slicing',
                                                       'hot, clean {obj} is placed',
                                                       'cold, sliced {obj} is placed',
                                                       'hot, sliced {obj} is cleaned',
                                                       '{obj} is placed after cleaning and heating',
                                                       '{obj} is placed after slicing and cooling']

###########################################################################################
###########################################################################################

gdict['clean_then_cool_and_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_then_cool_and_slice']['templates_pos'] = ['{obj} is cleaned, then cooled and sliced',
                                                       '{obj} is cooled and sliced after cleaning',
                                                       '{obj} is cleaned before cooling and slicing']
gdict['clean_then_cool_and_slice']['templates_neg'] = ['{obj} is cooled and sliced before cleaning',
                                                       '{obj} is sliced and cooled before cleaning',
                                                       '{obj} is cleaned after slicing and cooling',
                                                       'hot {obj} is cooled and sliced',
                                                       'clean {obj} is sliced and heated',
                                                       'a cold, sliced {obj} is cleaned',
                                                       'sliced, cold {obj} is cleaned']

gdict['clean_then_heat_and_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_then_heat_and_slice']['templates_pos'] = ['{obj} is cleaned, then heated and sliced',
                                                       '{obj} is heated and sliced after cleaning',
                                                       '{obj} is cleaned before heating and slicing']
gdict['clean_then_heat_and_slice']['templates_neg'] = ['{obj} is heated and sliced before cleaning',
                                                       '{obj} is sliced and heated before cleaning',
                                                       '{obj} is cleaned after slicing and heating',
                                                       'cold {obj} is heated and sliced',
                                                       'clean {obj} is cooled and sliced',
                                                       'clean {obj} is sliced and cooled',
                                                       'hot, sliced {obj} is cleaned',
                                                       'sliced, hot {obj} is cleaned']

gdict['cool_then_clean_and_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['cool_then_clean_and_slice']['templates_pos'] = ['{obj} is cooled, then cleaned and sliced',
                                                       '{obj} is cleaned and sliced after cooling',
                                                       '{obj} is cooled before cleaning and slicing']
gdict['cool_then_clean_and_slice']['templates_neg'] = ['{obj} is cleaned and sliced before cooling',
                                                       '{obj} is sliced and cleaned before cooling',
                                                       '{obj} is cooled after cleaning and slicing',
                                                       'hot {obj} is cleaned and sliced',
                                                       'sliced {obj} is cooled and cleaned',
                                                       'cold {obj} is cleaned and heated',
                                                       'sliced, clean {obj} is cooled',
                                                       'clean, sliced {obj} is cooled']

gdict['heat_then_clean_and_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['heat_then_clean_and_slice']['templates_pos'] = ['{obj} is heated, then cleaned and sliced',
                                                       '{obj} is cleaned and sliced after heating',
                                                       '{obj} is heated before cleaning and slicing']
gdict['heat_then_clean_and_slice']['templates_neg'] = ['{obj} is cleaned and sliced before heating',
                                                       '{obj} is sliced and cleaned before heating',
                                                       '{obj} is heated after cleaning and slicing',
                                                       'cold {obj} is cleaned and sliced',
                                                       'sliced {obj} is heated and cleaned',
                                                       'cold {obj} is cleaned and sliced',
                                                       'clean, sliced {obj} is heated',
                                                       'sliced, clean {obj} is heated']

gdict['slice_then_clean_and_cool']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['slice_then_clean_and_cool']['templates_pos'] = ['{obj} is sliced, then cleaned and cooled',
                                                       '{obj} is cleaned and cooled after slicing',
                                                       '{obj} is sliced before cleaning and cooling']
gdict['slice_then_clean_and_cool']['templates_neg'] = ['{obj} is cooled and cleaned before slicing',
                                                       '{obj} is sliced after cleaning and cooling',
                                                       'sliced {obj} is cleaned and heated',
                                                       'cold, clean {obj} is sliced',
                                                       'clean, cold {obj} is sliced',
                                                       'clean {obj} is sliced and cooled']

gdict['slice_then_clean_and_heat']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['slice_then_clean_and_heat']['templates_pos'] = ['{obj} is sliced, then cleaned and heated',
                                                       '{obj} is cleaned and heated after slicing',
                                                       '{obj} is sliced before cleaning and heating']
gdict['slice_then_clean_and_heat']['templates_neg'] = ['{obj} is heated and cleaned before slicing',
                                                       '{obj} is cleaned and heated before slicing',
                                                       '{obj} is sliced after cleaning and heating',
                                                       '{obj} is sliced after heating and cleaning',
                                                       'sliced {obj} is cleaned and cooled',
                                                       'sliced {obj} is cooled and cleaned',
                                                       'hot, clean {obj} is sliced',
                                                       'clean, hot {obj} is sliced',
                                                       'clean {obj} is sliced and heated']

# clean_then_cool_and_place, clean_then_heat_and_place, clean_then_slice_and_place, slice_then_cool_and_place
# slice_then_heat_and_place, slice_then_clean_and_place, heat_then_clean_and_place, heat_then_slice_and_place
# cool_then_clean_and_place, cool_then_slice_and_place
gdict['clean_then_cool_and_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['clean_then_cool_and_place']['templates_pos'] = ['{obj} is cleaned, then cooled and placed',
                                                       '{obj} is cooled and placed after cleaning',
                                                       '{obj} is cleaned before cooling and placing']
gdict['clean_then_cool_and_place']['templates_neg'] = ['{obj} is cooled and placed before cleaning',
                                                       '{obj} is cleaned after cooling and placing',
                                                       'clean {obj} is heated and placed',
                                                       'clean, hot {obj} is placed',
                                                       'clean, cold {obj} is sliced']

gdict['clean_then_heat_and_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['clean_then_heat_and_place']['templates_pos'] = ['{obj} is cleaned, then heated and placed',
                                                       '{obj} is heated and placed after cleaning',
                                                       '{obj} is cleaned before heating and placing']
gdict['clean_then_heat_and_place']['templates_neg'] = ['{obj} is heated and placed before cleaning',
                                                       '{obj} is cleaned after heating and placing',
                                                       'clean {obj} is cooled and placed',
                                                       'clean, cold {obj} is placed',
                                                       'clean, hot {obj} is sliced']

gdict['clean_then_slice_and_place']['pddl'] = gdict['slice_and_clean_and_place']['pddl']
gdict['clean_then_slice_and_place']['templates_pos'] = ['{obj} is cleaned, then sliced and placed',
                                                        '{obj} is sliced and placed after cleaning',
                                                        '{obj} is cleaned before slicing and placing']
gdict['clean_then_slice_and_place']['templates_neg'] = ['{obj} is sliced and placed before cleaning',
                                                        '{obj} is cleaned after slicing and placing',
                                                        'clean {obj} is cooled and placed',
                                                        'cold, sliced {obj} is placed',
                                                        'clean, sliced {obj} is heated']

gdict['slice_then_cool_and_place']['pddl'] = gdict['cool_and_slice_and_place']['pddl']
gdict['slice_then_cool_and_place']['templates_pos'] = ['{obj} is sliced, then cooled and placed',
                                                       '{obj} is cooled and placed after slicing',
                                                       '{obj} is sliced before cooling and placing']
gdict['slice_then_cool_and_place']['templates_neg'] = ['{obj} is cooled and placed before slicing',
                                                       '{obj} is sliced after cooling and placing',
                                                       'sliced {obj} is heated and placed',
                                                       'clean, sliced {obj} is placed',
                                                       'sliced, cold {obj} is cleaned']

gdict['slice_then_heat_and_place']['pddl'] = gdict['slice_and_heat_and_place']['pddl']
gdict['slice_then_heat_and_place']['templates_pos'] = ['{obj} is sliced, then heated and placed',
                                                       '{obj} is heated and placed after slicing',
                                                       '{obj} is sliced before heating and placing']
gdict['slice_then_heat_and_place']['templates_neg'] = ['{obj} is heated and placed before slicing',
                                                       '{obj} is sliced after heating and placing',
                                                       'sliced {obj} is cooled and placed',
                                                       'clean, hot {obj} is placed',
                                                       'sliced, hot {obj} is cleaned']

gdict['slice_then_clean_and_place']['pddl'] = gdict['slice_and_clean_and_place']['pddl']
gdict['slice_then_clean_and_place']['templates_pos'] = ['{obj} is sliced, then cleaned and placed',
                                                        '{obj} is cleaned and placed after slicing',
                                                        '{obj} is sliced before cleaning and placing']
gdict['slice_then_clean_and_place']['templates_neg'] = ['{obj} is cleaned and placed before slicing',
                                                        '{obj} is sliced after cleaning and placing',
                                                        'sliced {obj} is cooled and placed',
                                                        'hot, clean {obj} is placed',
                                                        'sliced, clean {obj} is heated']

gdict['heat_then_clean_and_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['heat_then_clean_and_place']['templates_pos'] = ['{obj} is heated, then cleaned and placed',
                                                       '{obj} is cleaned and placed after heating',
                                                       '{obj} is heated before cleaning and placing']
gdict['heat_then_clean_and_place']['templates_neg'] = ['{obj} is cleaned and placed before heating',
                                                       '{obj} is heated after cleaning and placing',
                                                       'cold {obj} is cleaned and placed',
                                                       'hot {obj} is sliced and placed',
                                                       'cold, clean {obj} is placed',
                                                       'hot, clean {obj} is sliced']

gdict['heat_then_slice_and_place']['pddl'] = gdict['slice_and_heat_and_place']['pddl']
gdict['heat_then_slice_and_place']['templates_pos'] = ['{obj} is heated, then sliced and placed',
                                                       '{obj} is sliced and placed after heating',
                                                       '{obj} is heated before slicing and placing']
gdict['heat_then_slice_and_place']['templates_neg'] = ['{obj} is sliced and placed before heating',
                                                       '{obj} is heated after slicing and placing',
                                                       'cold {obj} is sliced and placed',
                                                       'hot {obj} is cleaned and placed',
                                                       'cold, sliced {obj} is placed',
                                                       'a heat, sliced {obj} is cleaned']

gdict['cool_then_clean_and_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['cool_then_clean_and_place']['templates_pos'] = ['cold {obj} is cleaned and placed'
                                                       '{obj} is cooled, then cleaned and placed',
                                                       '{obj} is cleaned and placed after cooling',
                                                       '{obj} is cooled before cleaning and placing']
gdict['cool_then_clean_and_place']['templates_neg'] = ['{obj} is cleaned and placed before cooling',
                                                       '{obj} is cooled after cleaning and placing',
                                                       'hot {obj} is cleaned and placed',
                                                       'clean {obj} is cooled and placed',
                                                       'cold {obj} is cleaned and slice',
                                                       'cold, clean {obj} is sliced']

gdict['cool_then_slice_and_place']['pddl'] = gdict['cool_and_slice_and_place']['pddl']
gdict['cool_then_slice_and_place']['templates_pos'] = ['{obj} is cooled, then sliced and placed',
                                                       '{obj} is sliced and placed after cooling',
                                                       '{obj} is cooled before slicing and placing']
gdict['cool_then_slice_and_place']['templates_neg'] = ['{obj} is sliced and placed before cooling',
                                                       '{obj} is cooled after slicing and placing',
                                                       'hot {obj} is sliced and placed',
                                                       'sliced {obj} is cooled and placed',
                                                       'cold, sliced {obj} is cleaned']
