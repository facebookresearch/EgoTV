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

# basic pick and cool (e.g: "cool an apple")
gdict["cool_simple"] = \
    {
        'pddl':
            '''
        (:goal
            (and
                (exists (?o # object)
                    (exists (?a # agent)
                        (and
                            (coolable ?o)
                            (objectType ?o {obj}Type)
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
        'templates_pos': ['a cold {obj}',
                          '{obj} is cooled in a Fridge'],
        'templates_neg': ['a sliced {obj}',
                          '{obj} is heated',
                          '{obj} is cleaned in a SinkBasin']
    }

# basic pick and heat (e.g: "heat an apple")
gdict["heat_simple"] = \
    {
        'pddl':
            '''
        (:goal
            (and
                (exists (?o # object)
                    (exists (?a # agent)
                        (and
                            (heatable ?o)
                            (objectType ?o {obj}Type)
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
        'templates_pos': ['a hot {obj}',
                          '{obj} is heated',
                          '{obj} is picked and heated'],
        'templates_neg': ['a clean {obj} in a SinkBasin',
                          '{obj} is cooled in a Fridge',
                          'a sliced {obj}']
    }

# basic pick and clean (e.g: "clean an apple")
gdict["clean_simple"] = \
    {
        'pddl':
            '''
        (:goal
            (and
                (exists (?o # object)
                    (exists (?a # agent)
                        (and
                            (cleanable ?o)
                            (objectType ?o {obj}Type)
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
        'templates_pos': ['a clean {obj}',
                          '{obj} is cleaned',
                          '{obj} is cleaned in a SinkBasin'],
        'templates_neg': ['a hot {obj}',
                          '{obj} is cooled in a Fridge',
                          '{obj} is sliced']
    }

# basic locate and pick (e.g: "pick a apple")
gdict["pick_simple"] = \
    {
        'pddl':
            '''
        (:goal
            (and
                (exists (?o # object)
                    (exists (?a # agent)
                        (and
                            (holds ?a ?o)
                            (holdsAny ?a)
                        )
                    )
                )
            )
        )
    )
    ''',
        'templates_pos': ['{obj} is picked',
                          '{obj} is located and picked'],
        'templates_neg': ['{obj} is placed',
                          '{obj} is cooled in a Fridge',
                          '{obj} is cleaned']
    }

# basic slice task (e.g.: "slice an apple")
gdict["slice_simple"] = \
    {
        'pddl':
            '''
        (:goal
            (and
                (exists (?o # object)
                    (exists (?a # agent)
                        (and
                            (sliceable ?o)
                            (objectType ?o {obj}Type)
                            (isSliced ?o)
                            (holds ?a ?o)
                            (holdsAny ?a)
                        )                        
                    )                    
                )
            )
        )
    )
    ''',
        'templates_pos': ['a sliced {obj}',
                          '{obj} is sliced',
                          'a slice of {obj}'],
        'templates_neg': ['a cold {obj}',
                          '{obj} is heated',
                          '{obj} is cleaned']
    }

# toggle the state of a toggleable object (e.g: "toggle the lightswitch")
gdict["toggle_simple"] = \
    {
        'pddl':
            '''
        (:goal
             (and
                (exists (?ot # object)
                    (exists (?o # object)
                        (exists (?a # agent)
                            (and
                                (objectType ?o {obj}Type)
                                (objectType ?ot {toggle}Type)
                                (toggleable ?ot)
                                (isToggled ?ot) 
                                (holds ?a ?o)
                                (holdsAny ?a)
                            )
                        )
                    )
                )                 
             )
        )
    )
    ''',
        'templates_pos': ['{obj} is examined under {toggle}']
    }

# basic pick and place (e.g: "put the apple in the microwave")
gdict["place_simple"] = \
    {
        'pddl':
            '''
        (:goal
            (and
                (exists (?r # receptacle)
                    (exists (?o # object)
                        (and 
                            (inReceptacle ?o ?r) 
                            (objectType ?o {obj}Type) 
                            (receptacleType ?r {recep}Type)
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
        'templates_pos': ['{obj} placed in a {recep}',
                          '{obj} is picked and placed in a {recep}'],
        'templates_neg': ['{obj} is sliced',
                          '{obj} is picked and heated',
                          '{obj} is picked and cleaned']
    }

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
        'templates_pos': ['{obj} is cleaned in a SinkBasin and placed in a {recep}',
                          'a clean {obj} in a {recep}',
                          'a clean {obj} placed in a {recep}'],
        'templates_neg': ['{obj} is cooled in a Fridge and placed in a {recep}',
                          '{obj} is picked, heated and placed',
                          'a hot {obj} in a SinkBasin',
                          '{obj} is sliced in a {recep}']
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
        'templates_pos': ['a hot {obj} in a {recep}',
                          'a hot {obj} placed in a {recep}',
                          '{obj} is picked, heated and placed in a {recep}'],
        'templates_neg': ['{obj} is cleaned in a SinkBasin and placed in a {recep}',
                          '{obj} is picked, heated and cleaned',
                          'a cold {obj} in a {recep}']
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
        'templates_pos': ['a cold {obj} in a {recep}',
                          'a cold {obj} placed in a {recep}',
                          '{obj} cooled in a Fridge and placed in a {recep}'],
        'templates_neg': ['a hot {obj} in a {recep}',
                          '{obj} is cleaned and placed in a SinkBasin',
                          '{obj} is cooled in a Fridge and sliced']
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
        'templates_pos': ['a sliced {obj} in a {recep}',
                          'a sliced {obj} placed in a {recep}',
                          '{obj} is sliced and placed in a {recep}'],
        'templates_neg': ['a cold {obj} in a {recep}',
                          'a clean {obj} placed in a {recep}',
                          '{obj} is heated and placed in a {recep}']
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
        'templates_pos': ['{obj} is sliced and cooled in a Fridge',
                          '{obj} is cooled in a Fridge and sliced'],
        'templates_neg': ['a hot {obj} placed in a Fridge',
                          '{obj} is sliced and heated',
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
        'templates_pos': ['{obj} is heated and sliced',
                          '{obj} is sliced and heated'],
        'templates_neg': ['{obj} is cooled and sliced',
                          '{obj} is sliced and placed',
                          'a hot {obj} is cleaned']
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
        'templates_pos': ['{obj} is sliced and cleaned in a SinkBasin',
                          '{obj} is cleaned in a SinkBasin and sliced'],
        'templates_neg': ['{obj} is cooled and placed in a SinkBasin',
                          'a hot {obj} is sliced in a SinkBasin',
                          '{obj} is picked, heated and cleaned']
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
        'templates_pos': ['{obj} is cooled in a Fridge and cleaned in a SinkBasin',
                          '{obj} is cleaned in a SinkBasin and cooled in a Fridge'],
        'templates_neg': ['{obj} is cooled in a Fridge, sliced, and cleaned in a SinkBasin',
                          '{obj} is picked, cooled in a Fridge, and sliced in a SinkBasin']
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
        'templates_pos': ['{obj} is heated and cleaned in a SinkBasin',
                          '{obj} is cleaned in a SinkBasin and heated'],
        'templates_neg': ['{obj} is cleaned in a SinkBasin and sliced',
                          '{obj} is heated, cleaned and placed in a Fridge']
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
        'templates_pos': ['two {obj}s placed in a {recep}']
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
        'templates_pos': ['a hot {obj} in a {mrecep} in a {recep}',
                          'a hot {obj} in a {mrecep} placed in a {recep}',
                          'a {mrecep} containing a hot {obj} in a {recep}',
                          'a {mrecep} containing a hot {obj} placed in a {recep}']
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
        'templates_pos': ['a cold {obj} in a {mrecep} in a {recep}',
                          'a cold {obj} in a {mrecep} placed in a {recep}',
                          'a {mrecep} containing a cold {obj} in a {recep}',
                          'a {mrecep} containing a cold {obj} placed in a {recep}']
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
        'templates_pos': ['a clean {obj} in a {mrecep} in a {recep}',
                          'a clean {obj} in a {mrecep} placed in a {recep}',
                          'a {mrecep} containing a clean {obj} in a {recep}',
                          'a {mrecep} containing a clean {obj} placed in a {recep}']
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
        'templates_pos': ['a sliced {obj} in a {mrecep} in a {recep}',
                          'a sliced {obj} in a {mrecep} placed in a {recep}',
                          'a {mrecep} containing a sliced {obj} placed in a {recep}',
                          'a {mrecep} containing a sliced {obj} in a {recep}']
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
        'templates_pos': ['a clean, sliced {obj} placed in a {recep}',
                          'a sliced, clean {obj} placed in a {recep}',
                          '{obj} is sliced, cleaned in a SinkBasin and placed in a {recep}',
                          '{obj} is cleaned in a SinkBasin, sliced and placed in a {recep}'],
        'templates_neg': ['a sliced, cold {obj} placed in a {recep}',
                          'a sliced {obj} placed in a SinkBasin',
                          '{obj} is cleaned and placed in a SinkBasin',
                          'a clean {obj} is heated and placed in a {recep}',
                          '{obj} is picked, heated, sliced and placed in a {recep}']

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
        'templates_pos': ['{obj} is sliced, cleaned in a SinkBasin and heated',
                          '{obj} is heated, sliced and cleaned in a SinkBasin',
                          '{obj} is cleaned in a SinkBasin, sliced and heated'],
        'templates_neg': ['{obj} is cooled in a Fridge, sliced and cleaned in a SinkBasin',
                          'a cold, clean, sliced {obj}',
                          '{obj} is sliced, heated and placed in a SinkBasin']
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
        'templates_pos': ['a clean, hot {obj} placed in a {recep}',
                          'a hot, clean {obj} placed in a {recep}',
                          '{obj} is cleaned in a SinkBasin, heated and placed in a {recep}',
                          '{obj} is heated, cleaned in a SinkBasin and placed in a {recep}'],
        'templates_neg': ['a cool, clean {obj} is sliced',
                          'a hot, clean, sliced {obj}',
                          '{obj} is cleaned in a SinkBasin, cooled and placed in a {recep}',
                          '{obj} is picked, cooled and placed in a SinkBasin']
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
        'templates_pos': ['{obj} is cooled, sliced and cleaned',
                          '{obj} is cooled in a Fridge, cleaned in a SinkBasin and sliced',
                          '{obj} is cleaned, sliced and cooled',
                          '{obj} is cleaned in a SinkBasin, sliced and cooled in a Fridge',
                          '{obj} is sliced, cooled in a Fridge, and cleaned in a SinkBasin',
                          '{obj} is sliced, cleaned in a SinkBasin, and cooled in a Fridge'],
        'templates_neg': ['a hot, sliced {obj} placed in a Fridge',
                          '{obj} is picked, heated and placed in a SinkBasin',
                          'a hot, clean, sliced {obj}']
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
        'templates_pos': ['a clean, cold {obj} placed in a {recep}',
                          'a clean, cold {obj} placed in a {recep}',
                          '{obj} is cooled, cleaned, and placed in a {recep}',
                          '{obj} is cleaned in a SinkBasin, cooled in a Fridge, and placed in a {recep}',
                          '{obj} is cleaned, cooled and placed in a {recep}',
                          '{obj} is cooled in a Fridge, cleaned in a SinkBasin, and placed in a {recep}'],
        'templates_neg': ['a hot, clean {obj} placed in a {recep}',
                          'a cold, sliced {obj} in a {recep}',
                          'a clean, cold {obj} sliced in a {recep}',
                          '{obj} is cooled in a Fridge, cleaned in a SinkBasin, and sliced in a {recep}']
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
        'templates_pos': ['a hot, sliced {obj} placed in a {recep}',
                          'a sliced, hot {obj} placed in a {recep}',
                          '{obj} is sliced, heated and placed in a {recep}',
                          '{obj} is heated, sliced and placed in a {recep}',
                          '{obj} is heated, placed in a {recep} and sliced'],
        'templates_neg': ['a cold, sliced {obj} placed in a {recep}',
                          '{obj} is picked, heated, sliced and cleaned in a {recep}',
                          '{obj} is sliced, cleaned and cooled in a Fridge']
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
        'templates_pos': ['a cold, sliced {obj} placed in a {recep}',
                          'a sliced, cold {obj} placed in a {recep}',
                          '{obj} is cooled in a Fridge, sliced, and placed in a {recep}',
                          '{obj} is sliced, cooled in a Fridge, and placed in a {recep}'],
        'templates_neg': ['a hot, sliced {obj} placed in a {recep}',
                          'a cold, sliced {obj} cleaned in a {recep}',
                          '{obj} is picked, cooled in a Fridge, sliced and heated']
    }

# pick two instances of a sliced object and place them in a receptacle (e.g: "pick two apples and put them in the sink")
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
        'templates_pos': ['Provide a hot {mrecep} of {obj} in {recep}']
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
        'templates_pos': ['Provide a cold {mrecep} of {obj} in {recep}']
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
        'templates_pos': ['Provide a cold {mrecep} of {obj} in {recep}']
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
        'templates_pos': ['{obj} is cleaned in a SinkBasin, then sliced, then placed in a {recep}'],
        'templates_neg': ['{obj} is sliced, then cleaned in a SinkBasin, then placed in a {recep}',
                          '{obj} is sliced before cleaning in a SinkBasin and placing in a {recep}',
                          '{obj} is cleaned in a SinkBasin after slicing and placing in a {recep}']

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
        'templates_pos': ['{obj} is heated, then sliced, then placed in a {recep}',
                          '{obj} is picked, then heated, then sliced, then placed in a {recep}'],
        'template_neg': ['{obj} is sliced, then heated, then placed in a {recep}',
                         '{obj} is sliced before heating and placing in a {recep}',
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
        'templates_pos': ['{obj} is cooled in a Fridge, then sliced, then placed in a {recep}'],
        'template_neg': ['{obj} is sliced, then cooled in a Fridge, then placed in a {recep}',
                         '{obj} is sliced before cooling in a Fridge and placing in a {recep}',
                         '{obj} is picked and cooled in a Fridge after slicing']
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
gdict['clean_then_cool']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then cooled in a Fridge',
                                             '{obj} is cleaned in a SinkBasin before cooling in a Fridge',
                                             '{obj} is cooled in a Fridge after cleaning in a SinkBasin']
gdict['clean_then_cool']['templates_neg'] = ['a clean {obj} is heated',
                                             'a hot {obj} is cooled in a Fridge',
                                             'a cold {obj} is cleaned in a SinkBasin',
                                             'a clean, cold {obj} is sliced',
                                             '{obj} is cooled in a Fridge, then cleaned in a SinkBasin',
                                             '{obj} is cleaned in a SinkBasin after cooling in a Fridge',
                                             '{obj} is cooled in a Fridge after cleaning in a SinkBasin']
gdict['cool_then_clean']['pddl'] = gdict['cool_and_clean']['pddl']
gdict['cool_then_clean']['templates_pos'] = ['{obj} is cooled in a Fridge, then cleaned in a SinkBasin',
                                             '{obj} is cleaned in a SinkBasin after cooling in a Fridge',
                                             '{obj} is cooled in a Fridge after cleaning in a SinkBasin']
gdict['cool_then_clean']['templates_neg'] = ['a hot {obj} is cleaned in a SinkBasin',
                                             'a clean {obj} is cooled in a Fridge',
                                             'a cold, clean {obj} is sliced',
                                             '{obj} is cooled in a Fridge, then sliced in a SinkBasin',
                                             '{obj} is cleaned in a SinkBasin, then cooled in a Fridge',
                                             '{obj} is cleaned in a SinkBasin before cooling in a Fridge',
                                             '{obj} is cooled in a Fridge after cleaning in a SinkBasin']

gdict['clean_then_heat']['pddl'] = gdict['clean_and_heat']['pddl']
gdict['clean_then_heat']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then heated',
                                             '{obj} is cleaned in a SinkBasin before heating',
                                             '{obj} is heated after cleaning in a SinkBasin']
gdict['clean_then_heat']['templates_neg'] = ['a clean {obj} is cooled in a Fridge',
                                             'a hot {obj} is cleaned in a SinkBasin',
                                             'a sliced {obj} is heated',
                                             'a clean, hot {obj} is sliced',
                                             '{obj} is heated, then cleaned in a SinkBasin',
                                             '{obj} is cleaned in a SinkBasin after heating',
                                             '{obj} is picked and heated before cleaning in a SinkBasin']
gdict['heat_then_clean']['pddl'] = gdict['clean_and_heat']['pddl']
gdict['heat_then_clean']['templates_pos'] = ['{obj} is heated, then cleaned in a SinkBasin',
                                             '{obj} is cleaned in a SinkBasin after heating',
                                             '{obj} is picked and heated before cleaning in a SinkBasin']
gdict['heat_then_clean']['templates_neg'] = ['a clean {obj} is heated',
                                             'a hot {obj} is placed and sliced in a SinkBasin',
                                             'a cold {obj} is cleaned in a SinkBasin',
                                             'a hot, clean {obj} is sliced',
                                             '{obj} is cleaned in a SinkBasin, then heated',
                                             '{obj} is cleaned in a SinkBasin before heating',
                                             '{obj} is heated after cleaning in a SinkBasin']

gdict['cool_then_slice']['pddl'] = gdict['cool_and_slice']['pddl']
gdict['cool_then_slice']['templates_pos'] = ['{obj} is cooled in a Fridge, then sliced',
                                             '{obj} is cooled in a Fridge before slicing',
                                             '{obj} is sliced after cooling in a Fridge']
gdict['cool_then_slice']['templates_neg'] = ['a cold {obj} is cleaned in a SinkBasin',
                                             'a hot {obj} is sliced',
                                             'a cold, sliced {obj} is cleaned in a SinkBasin',
                                             'a sliced {obj} is cooled in a Fridge',
                                             '{obj} is sliced, then cooled in a Fridge',
                                             '{obj} is sliced before cooling in a Fridge',
                                             '{obj} is cooled in a Fridge after slicing']
gdict['slice_then_cool']['pddl'] = gdict['cool_and_slice']['pddl']
gdict['slice_then_cool']['templates_pos'] = ['{obj} is sliced, then cooled in a Fridge',
                                             '{obj} is sliced before cooling in a Fridge',
                                             '{obj} is cooled in a Fridge after slicing']
gdict['slice_then_cool']['templates_neg'] = ['a cold {obj} is sliced',
                                             'a sliced {obj} is heated',
                                             'a cold, sliced {obj} is cleaned in a SinkBasin',
                                             '{obj} is cooled in a Fridge, then sliced',
                                             '{obj} is cooled in a Fridge before slicing',
                                             '{obj} is sliced after cooling in a Fridge']

gdict['clean_then_slice']['pddl'] = gdict['clean_and_slice']['pddl']
gdict['clean_then_slice']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then sliced',
                                              '{obj} is cleaned in a SinkBasin before slicing',
                                              '{obj} is sliced after cleaning in a SinkBasin']
gdict['clean_then_slice']['templates_neg'] = ['a cold {obj} is sliced',
                                              'a sliced {obj} is cleaned in a SinkBasin',
                                              'a clean, sliced {obj} is cooled in a SinkBasin',
                                              '{obj} is sliced, then cleaned in a SinkBasin',
                                              '{obj} is sliced before cleaning in a SinkBasin',
                                              '{obj} is cleaned in a SinkBasin after slicing']
gdict['slice_then_clean']['pddl'] = gdict['clean_and_slice']['pddl']
gdict['slice_then_clean']['templates_pos'] = ['{obj} is sliced, then cleaned in a SinkBasin',
                                              '{obj} is sliced before cleaning in a SinkBasin',
                                              '{obj} is cleaned in a SinkBasin after slicing']
gdict['slice_then_clean']['templates_neg'] = ['a clean {obj} is sliced',
                                              'a sliced {obj} is cooled in a Fridge',
                                              'a clean, sliced {obj} is cooled in a SinkBasin',
                                              '{obj} is cleaned in a SinkBasin, then sliced',
                                              '{obj} is cleaned in a SinkBasin before slicing',
                                              '{obj} is sliced after cleaning in a SinkBasin']

gdict['heat_then_slice']['pddl'] = gdict['heat_and_slice']['pddl']
gdict['heat_then_slice']['templates_pos'] = ['{obj} is picked, heated, then sliced',
                                             '{obj} is picked and heated before slicing',
                                             '{obj} is sliced after heating']
gdict['heat_then_slice']['templates_neg'] = ['a sliced {obj} is heated',
                                             'a cold {obj} is sliced',
                                             'a hot {obj} is cleaned in a SinkBasin',
                                             '{obj} is sliced, then heated',
                                             '{obj} is sliced before heating',
                                             '{obj} is heated after slicing']
gdict['slice_then_heat']['pddl'] = gdict['heat_and_slice']['pddl']
gdict['slice_then_heat']['templates_pos'] = ['{obj} is sliced, then heated',
                                             '{obj} is sliced before heating',
                                             '{obj} is heated after slicing']
gdict['slice_then_heat']['templates_neg'] = ['a sliced {obj} cooled in a Fridge',
                                             'a hot {obj} is sliced',
                                             'a hot, sliced {obj} is cleaned in a SinkBasin',
                                             '{obj} is picked, heated, then sliced',
                                             '{obj} is picked and heated before slicing',
                                             '{obj} is sliced after heating']

gdict['slice_then_place']['pddl'] = gdict['slice_and_place']['pddl']
gdict['slice_then_place']['templates_pos'] = ['{obj} is sliced, then placed in a {recep}',
                                              '{obj} is sliced before placing in a {recep}',
                                              '{obj} is placed in a {recep} after slicing']
gdict['slice_then_place']['templates_neg'] = ['a clean {obj} is placed in a {recep}',
                                              'a sliced {obj} is cooled and placed in a {recep}',
                                              '{obj} is heated, then placed in a {recep}',
                                              '{obj} is cooled in a Fridge before placing in a {recep}',
                                              '{obj} is placed in a {recep} after cleaning in a SinkBasin']

gdict['clean_then_place']['pddl'] = gdict['clean_and_place']['pddl']
gdict['clean_then_place']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then placed in a {recep}',
                                              '{obj} is cleaned in a SinkBasin before placing in a {recep}',
                                              '{obj} is placed in a {recep} after cleaning in a SinkBasin']
gdict['clean_then_place']['templates_neg'] = ['a cold {obj} is placed in a {recep}',
                                              'a clean {obj} is sliced and placed in a {recep}',
                                              '{obj} is heated, then placed in a {recep}',
                                              '{obj} is cooled in a Fridge before placing in a {recep}',
                                              '{obj} is placed in a {recep} after slicing']

gdict['cool_then_place']['pddl'] = gdict['cool_and_place']['pddl']
gdict['cool_then_place']['templates_pos'] = ['{obj} is cooled in a Fridge, then placed in a {recep}',
                                             '{obj} is cooled in a Fridge before placing in a {recep}',
                                             '{obj} is placed in a {recep} after cooling in a Fridge']
gdict['cool_then_place']['templates_neg'] = ['a hot {obj} is placed in a {recep}',
                                             'a cold {obj} is sliced and placed in a {recep}',
                                             '{obj} is heated, then placed in a {recep}',
                                             '{obj} is cleaned in a SinkBasin before placing in a {recep}',
                                             '{obj} is placed in a {recep} after slicing']

gdict['heat_then_place']['pddl'] = gdict['heat_and_place']['pddl']
gdict['heat_then_place']['templates_pos'] = ['{obj} is heated, then placed in a {recep}',
                                             '{obj} is heated before placing in a {recep}',
                                             '{obj} is placed in a {recep} after heating',
                                             '{obj} is picked and heated before placing in a {recep}',
                                             '{obj} is picked, heated, then placed in a {recep}']
gdict['heat_then_place']['templates_neg'] = ['a cold {obj} is placed in a {recep}',
                                             'a hot {obj} is cleaned and placed in a {recep}',
                                             '{obj} is cooled, then placed in a {recep}',
                                             '{obj} is cleaned in a SinkBasin before placing in a {recep}',
                                             '{obj} is placed in a {recep} after slicing']

# ['cool_and_slice_and_place', 'cool_and_clean_and_place', 'cool_and_slice_and_clean', 'slice_and_heat_and_clean',
# 'slice_and_heat_and_place', 'slice_and_clean_and_place', 'heat_and_clean_and_place']
###########################################################################################
###########################################################################################

gdict['clean_then_cool_then_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['clean_then_cool_then_place']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then cooled in a Fridge, '
                                                        'then placed in a {recep}']
gdict['clean_then_cool_then_place']['templates_neg'] = ['{obj} is cooled in a Fridge, then cleaned in a SinkBasin, '
                                                        'then placed in a {recep}',
                                                        '{obj} is cleaned in SinkBasin after cooling in a Fridge',
                                                        '{obj} is cooled in a Fridge before cleaning in a SinkBasin '
                                                        'and placing in a {recep}']

gdict['clean_then_cool_then_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_then_cool_then_slice']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then cooled in a Fridge, '
                                                        'then sliced']
gdict['clean_then_cool_then_slice']['templates_neg'] = ['{obj} is cleaned in a SinkBasin, then sliced, '
                                                        'then cooled in a Fridge',
                                                        '{obj} is cooled in a Fridge, then cleaned in a SinkBasin, '
                                                        'then sliced',
                                                        '{obj} is sliced, then cleaned in a SinkBasin, then '
                                                        'cooled in a Fridge',
                                                        '{obj} is cooled in a Fridge and sliced,'
                                                        ' then cleaned in a SinkBasin',
                                                        '{obj} is cleaned in a SinkBasin after '
                                                        'cooling in a Fridge and slicing',
                                                        '{obj} is cooled in Fridge and sliced before '
                                                        'cleaning in SinkBasin']

gdict['clean_then_heat_then_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['clean_then_heat_then_place']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then heated, '
                                                        'then placed in a {recep}']
gdict['clean_then_heat_then_place']['templates_neg'] = ['{obj} is heated, then cleaned in a SinkBasin, '
                                                        'then placed in a {recep}',
                                                        '{obj} is picked, then heated, then cleaned in a SinkBasin, '
                                                        'then placed in a {recep}',
                                                        '{obj} is cleaned in a SinkBasin after heating',
                                                        '{obj} is heated before cleaning in a SinkBasin and placing '
                                                        'in a {recep}']

gdict['clean_then_heat_then_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_then_heat_then_slice']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then heated, '
                                                        'then sliced']
gdict['clean_then_heat_then_slice']['templates_neg'] = ['{obj} is heated, then cleaned in a SinkBasin, '
                                                        'then sliced',
                                                        '{obj} is picked, then heated, then cleaned in a SinkBasin, '
                                                        'then sliced',
                                                        '{obj} is sliced, then heated, then cleaned '
                                                        'in a SinkBasin',
                                                        '{obj} is heated, then sliced, then cleaned '
                                                        'in a SinkBasin',
                                                        '{obj} is picked, then heated, then sliced, then cleaned '
                                                        'in a SinkBasin',
                                                        '{obj} is sliced, then cleaned in a SinkBasin, '
                                                        'then heated',
                                                        '{obj} is heated after cleaning in a SinkBasin and slicing',
                                                        '{obj} cleaned in a SinkBasin and sliced before heating',
                                                        '{obj} is cleaned in a SinkBasin and heated '
                                                        'after slicing',
                                                        '{obj} is sliced before cleaning in a SinkBasin and heating']

gdict['clean_then_slice_then_cool']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_then_slice_then_cool']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then sliced, '
                                                        'then cooled in a Fridge']
gdict['clean_then_slice_then_cool']['templates_neg'] = ['{obj} is cooled in a Fridge, then cleaned in a SinkBasin, '
                                                        'then sliced',
                                                        '{obj} is cleaned in a SinkBasin, and cooled in a Fridge,'
                                                        ' then sliced',
                                                        '{obj} is sliced after cleaning in a SinkBasin and '
                                                        'cooling in a Fridge',
                                                        '{obj} is cleaned in a SinkBasin and cooled in a Fridge '
                                                        'before slicing',
                                                        '{obj} is sliced, then cleaned in a SinkBasin, then '
                                                        'cooled in a Fridge',
                                                        '{obj} is cleaned in a SinkBasin after cooling in a Fridge '
                                                        'and slicing',
                                                        '{obj} is cooled in a Fridge and sliced before cleaning '
                                                        'in a SinkBasin']

gdict['clean_then_slice_then_heat']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_then_slice_then_heat']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then sliced, '
                                                        'then heated']
gdict['clean_then_slice_then_heat']['templates_neg'] = ['{obj} is cleaned in a SinkBasin, then heated, '
                                                        'then sliced',
                                                        '{obj} is heated, then cleaned in a SinkBasin, '
                                                        'then sliced',
                                                        '{obj} is picked, then heated, then cleaned in a SinkBasin, '
                                                        'then sliced',
                                                        '{obj} is sliced, then heated, then cleaned '
                                                        'in a SinkBasin',
                                                        '{obj} is heated, then sliced, then cleaned '
                                                        'in a SinkBasin',
                                                        '{obj} is picked, then heated, then sliced, then cleaned '
                                                        'in a SinkBasin',
                                                        '{obj} is sliced, then cleaned in a SinkBasin, '
                                                        'then heated',
                                                        '{obj} is cleaned in a SinkBasin and heated '
                                                        'after slicing',
                                                        '{obj} is sliced before cleaning in a SinkBasin and heating',
                                                        '{obj} is sliced and cleaned in a SinkBasin after heating',
                                                        '{obj} is heated before cleaning in a SinkBasin and slicing']

gdict['cool_then_clean_then_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['cool_then_clean_then_place']['templates_pos'] = ['{obj} is cooled in a Fridge, then cleaned in a SinkBasin, '
                                                        'then placed in a {recep}']
gdict['cool_then_clean_then_place']['templates_neg'] = ['{obj} is cleaned in a SinkBasin, then cooled in a Fridge, '
                                                        'then placed in a {recep}',
                                                        '{obj} is cooled in a Fridge after cleaning in a SinkBasin',
                                                        '{obj} is cleaned in a SinkBasin before cooling in a Fridge '
                                                        'and placing in a {recep}']

gdict['cool_then_clean_then_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['cool_then_clean_then_slice']['templates_pos'] = ['{obj} is cooled in a Fridge, then cleaned in a SinkBasin, '
                                                        'then sliced']
gdict['cool_then_clean_then_slice']['templates_neg'] = ['{obj} is cleaned in a SinkBasin, then cooled in a Fridge, '
                                                        'then sliced',
                                                        '{obj} is cleaned in a SinkBasin, then sliced, '
                                                        'then cooled in a Fridge',
                                                        '{obj} is sliced, then cleaned in a SinkBasin, then '
                                                        'cooled in a Fridge',
                                                        '{obj} is cleaned in a SinkBasin, and cooled in a Fridge,'
                                                        ' then sliced',
                                                        '{obj} is cleaned in a SinkBasin after cooling in a Fridge '
                                                        'and slicing',
                                                        '{obj} is cooled in a Fridge and sliced before cleaning '
                                                        'in a SinkBasin',
                                                        '{obj} is cooled in a Fridge after cleaning in a SinkBasin '
                                                        'and slicing',
                                                        '{obj} is cleaned in a SinkBasin and sliced before cooling '
                                                        'in a Fridge']

gdict['cool_then_slice_then_clean']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['cool_then_slice_then_clean']['templates_pos'] = ['{obj} is cooled in a Fridge, then sliced, then cleaned '
                                                        'in a SinkBasin']
gdict['cool_then_slice_then_clean']['templates_neg'] = ['{obj} is cooled in a Fridge, then cleaned in a SinkBasin, '
                                                        'then sliced',
                                                        '{obj} is cleaned in a SinkBasin, then cooled in a Fridge, '
                                                        'then sliced',
                                                        '{obj} is cleaned in a SinkBasin, then sliced, '
                                                        'then cooled in a Fridge',
                                                        '{obj} is sliced, then cleaned in a SinkBasin then '
                                                        'cooled in a Fridge',
                                                        '{obj} is cooled in a Fridge after cleaning in a SinkBasin '
                                                        'and slicing',
                                                        '{obj} is cleaned in a SinkBasin and sliced before cooling '
                                                        'in a Fridge',
                                                        '{obj} is sliced after cooling in a Fridge and '
                                                        'cleaning in a SinkBasin',
                                                        '{obj} is cooled in a Fridge and cleaned in a SinkBasin '
                                                        'before slicing']

gdict['heat_then_clean_then_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['heat_then_clean_then_place']['templates_pos'] = ['{obj} is heated, then cleaned in a SinkBasin, '
                                                        'then placed in a {recep}',
                                                        '{obj} is picked, then heated, then cleaned in a SinkBasin, '
                                                        'then placed in a {recep}']
gdict['heat_then_clean_then_place']['templates_neg'] = ['{obj} is cleaned in a SinkBasin, then heated, '
                                                        'then placed in a {recep}',
                                                        '{obj} is picked, then cleaned in a SinkBasin, then heated, '
                                                        'then placed in a {recep}',
                                                        '{obj} is heated after cleaning in a SinkBasin',
                                                        '{obj} is cleaned in a SinkBasin before heating and placing '
                                                        'in a {recep}']

gdict['heat_then_clean_then_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['heat_then_clean_then_slice']['templates_pos'] = ['{obj} is heated, then cleaned in a SinkBasin, '
                                                        'then sliced',
                                                        '{obj} is picked, then heated, then cleaned in a SinkBasin, '
                                                        'then sliced']
gdict['heat_then_clean_then_slice']['templates_neg'] = ['{obj} is cleaned in a SinkBasin, then heated, '
                                                        'then sliced',
                                                        '{obj} is picked, then cleaned in a SinkBasin, then heated, '
                                                        'then sliced',
                                                        '{obj} is sliced, then heated, then cleaned '
                                                        'in a SinkBasin',
                                                        '{obj} is heated, then sliced, then cleaned '
                                                        'in a SinkBasin',
                                                        '{obj} is picked, then heated, then sliced, then cleaned '
                                                        'in a SinkBasin',
                                                        '{obj} is sliced, then cleaned in a SinkBasin '
                                                        'then heated',
                                                        '{obj} is heated after cleaning in a SinkBasin and slicing',
                                                        '{obj} cleaned in a SinkBasin and sliced before heating',
                                                        '{obj} is cleaned in a SinkBasin and heated '
                                                        'after slicing',
                                                        '{obj} is sliced before cleaning in a SinkBasin and heating']

gdict['heat_then_slice_then_clean']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['heat_then_slice_then_clean']['templates_pos'] = ['{obj} is heated, then sliced, then cleaned '
                                                        'in a SinkBasin',
                                                        '{obj} is picked, then heated, then sliced, then cleaned '
                                                        'in a SinkBasin']
gdict['heat_then_slice_then_clean']['templates_neg'] = ['{obj} is cleaned in a SinkBasin, then heated, '
                                                        'then sliced',
                                                        '{obj} is picked, then cleaned in a SinkBasin, then heated, '
                                                        'then sliced',
                                                        '{obj} is sliced, then heated, then cleaned '
                                                        'in a SinkBasin',
                                                        '{obj} is sliced, then cleaned in a SinkBasin, '
                                                        'then heated',
                                                        '{obj} is heated after cleaning in a SinkBasin and slicing',
                                                        '{obj} cleaned in a SinkBasin and sliced before heating',
                                                        '{obj} is cleaned in a SinkBasin before heating and slicing',
                                                        '{obj} is heated and sliced after cleaning in a SinkBasin']

gdict['slice_then_clean_then_cool']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['slice_then_clean_then_cool']['templates_pos'] = ['{obj} is sliced, then cleaned, then cooled',
                                                        '{obj} is sliced, then cleaned in a SinkBasin, then '
                                                        'cooled in a Fridge']
gdict['slice_then_clean_then_cool']['templates_neg'] = ['{obj} is cooled in a Fridge, then cleaned in a SinkBasin, '
                                                        'then sliced',
                                                        '{obj} is cleaned in a SinkBasin, then cooled in a Fridge, '
                                                        'then sliced',
                                                        '{obj} is cleaned in a SinkBasin, then sliced, '
                                                        'then cooled in a Fridge',
                                                        '{obj} is sliced, then cooled in a Fridge, then cleaned '
                                                        'in a SinkBasin',
                                                        '{obj} is sliced after cooling in a Fridge and cleaning in a '
                                                        'SinkBasin',
                                                        '{obj} is cleaned in a SinkBasin and cooled in a Fridge '
                                                        'before slicing',
                                                        '{obj} is cooled in Fridge before cleaning in a SinkBasin '
                                                        'and slicing',
                                                        '{obj} is cleaned in a SinkBasin and sliced after '
                                                        'cooling in a Fridge']

gdict['slice_then_clean_then_heat']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['slice_then_clean_then_heat']['templates_pos'] = ['{obj} is sliced, then cleaned in a SinkBasin, '
                                                        'then heated']
gdict['slice_then_clean_then_heat']['templates_neg'] = ['{obj} is heated, then cleaned in a SinkBasin, '
                                                        'then sliced',
                                                        '{obj} is cleaned in a SinkBasin, then heated, '
                                                        'then sliced',
                                                        '{obj} is cleaned in a SinkBasin, then sliced, '
                                                        'then heated',
                                                        '{obj} is sliced, then heated, then cleaned '
                                                        'in a SinkBasin',
                                                        '{obj} is sliced after cleaning in a SinkBasin and heating',
                                                        '{obj} is cleaned in a SinkBasin and heated '
                                                        'before slicing',
                                                        '{obj} is heated before cleaning in a SinkBasin '
                                                        'and slicing',
                                                        '{obj} is cleaned in a SinkBasin and sliced after '
                                                        'heating']

gdict['slice_then_clean_then_place']['pddl'] = gdict['slice_and_clean_and_place']['pddl']
gdict['slice_then_clean_then_place']['templates_pos'] = ['{obj} is sliced, then cleaned in a SinkBasin and '
                                                         'then placed in a {recep}']
gdict['slice_then_clean_then_place']['templates_neg'] = ['{obj} is cleaned in a SinkBasin, then sliced, '
                                                         'then placed in a {recep}',
                                                         '{obj} is sliced after cleaning in a SinkBasin',
                                                         '{obj} is cleaned in a SinkBasin before slicing and placing '
                                                         'in a {recep}']

gdict['slice_then_cool_then_clean']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['slice_then_cool_then_clean']['templates_pos'] = ['{obj} is sliced, then cooled in a Fridge, then cleaned '
                                                        'in a SinkBasin']
gdict['slice_then_cool_then_clean']['templates_neg'] = ['{obj} is cooled in a Fridge, then cleaned in a SinkBasin, '
                                                        'then sliced',
                                                        '{obj} is cleaned in a SinkBasin, then cooled in a Fridge, '
                                                        'then sliced',
                                                        '{obj} is cleaned in a SinkBasin, then sliced, '
                                                        'then cooled in a Fridge',
                                                        '{obj} is sliced, then cleaned in a SinkBasin, then '
                                                        'cooled in a Fridge',
                                                        '{obj} is sliced after cooling in a Fridge and cleaning in a '
                                                        'SinkBasin',
                                                        '{obj} is cleaned in a SinkBasin and cooled in a Fridge '
                                                        'before slicing',
                                                        '{obj} is cleaned before cooling in a Fridge '
                                                        'and slicing',
                                                        '{obj} is cooled in a Fridge and sliced after '
                                                        'cleaning in a SinkBasin']

gdict['slice_then_cool_then_place']['pddl'] = gdict['cool_and_slice_and_place']['pddl']
gdict['slice_then_cool_then_place']['templates_pos'] = ['{obj} is sliced, then cooled in a Fridge, then placed '
                                                        'in a {recep}']
gdict['slice_then_cool_then_place']['templates_neg'] = ['{obj} is cooled in a Fridge, then sliced, '
                                                        'then placed in a {recep}',
                                                        '{obj} is sliced after cooling in a Fridge',
                                                        '{obj} is cooled in a Fridge before slicing and placing '
                                                        'in a {recep}']

gdict['slice_then_heat_then_clean']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['slice_then_heat_then_clean']['templates_pos'] = ['{obj} is sliced, then heated, then cleaned '
                                                        'in a SinkBasin']
gdict['slice_then_heat_then_clean']['templates_neg'] = ['{obj} is heated, then cleaned in a SinkBasin, '
                                                        'then sliced',
                                                        '{obj} is cleaned in a SinkBasin, then heated, '
                                                        'then sliced',
                                                        '{obj} is cleaned in a SinkBasin, then sliced, '
                                                        'then heated',
                                                        '{obj} is sliced, then cleaned in a SinkBasin, then '
                                                        'heated',
                                                        '{obj} is sliced after heating and cleaning in a '
                                                        'SinkBasin',
                                                        '{obj} is cleaned in a SinkBasin and heated '
                                                        'before slicing',
                                                        '{obj} is cleaned before heating '
                                                        'and slicing',
                                                        '{obj} is heated and sliced after '
                                                        'cleaning in a SinkBasin']

gdict['slice_then_heat_then_place']['pddl'] = gdict['slice_and_heat_and_place']['pddl']
gdict['slice_then_heat_then_place']['templates_pos'] = ['{obj} is sliced, then heated, then placed in a {recep}']
gdict['slice_then_heat_then_place']['templates_neg'] = ['{obj} is heated, then sliced, '
                                                        'then placed in a {recep}',
                                                        '{obj} is sliced after heating',
                                                        '{obj} is heated before slicing and placing '
                                                        'in a {recep}']

###########################################################################################
###########################################################################################

gdict['clean_and_cool_then_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['clean_and_cool_then_place']['templates_pos'] = ['{obj} is cleaned in a SinkBasin and cooled in a Fridge,'
                                                       ' then placed in a {recep}',
                                                       '{obj} is cooled in a Fridge and cleaned in a SinkBasin,'
                                                       ' then placed in a {recep}',
                                                       '{obj} is placed in a {recep} after cleaning in a SinkBasin '
                                                       'and cooling in a Fridge',
                                                       '{obj} is placed in a {recep} after cooling in a Fridge and '
                                                       'cleaning in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin and cooled in a Fridge '
                                                       'before placing in a {recep}',
                                                       '{obj} is cooled in a Fridge and cleaned in a SinkBasin '
                                                       'before placing in a {recep}']
gdict['clean_and_cool_then_place']['templates_neg'] = ['{obj} is cleaned in a SinkBasin and heated before '
                                                       'placing in a {recep}',
                                                       '{obj} is cooled in a Fridge and heated before '
                                                       'placing in a {recep}',
                                                       '{obj} is cleaned in a SinkBasin and cooled in a Fridge '
                                                       'before slicing',
                                                       'a clean, sliced {obj} is placed in a {recep}',
                                                       'a hot, clean {obj} is placed in a {recep}',
                                                       'a cold, clean {obj} is sliced in a {recep}',
                                                       '{obj} is placed in a {recep} after cleaning in a SinkBasin '
                                                       'and heating',
                                                       '{obj} is placed in a {recep} after slicing '
                                                       'and heating']

gdict['clean_and_cool_then_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_and_cool_then_slice']['templates_pos'] = ['{obj} is cleaned in a SinkBasin and cooled in a Fridge,'
                                                       ' then sliced',
                                                       '{obj} is cooled in a Fridge and cleaned in a SinkBasin,'
                                                       ' then sliced',
                                                       '{obj} is sliced after cleaning in a SinkBasin and '
                                                       'cooling in a Fridge',
                                                       '{obj} is sliced after cooling in Fridge and '
                                                       'cleaning in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin and cooled in a Fridge '
                                                       'before slicing',
                                                       '{obj} is cooled in a Fridge and cleaned in a SinkBasin '
                                                       'before slicing']
gdict['clean_and_cool_then_slice']['templates_neg'] = ['{obj} is sliced before cleaning in a SinkBasin and '
                                                       'cooling in a Fridge',
                                                       '{obj} is cleaned in a SinkBasin and cooled in a Fridge '
                                                       'after slicing',
                                                       '{obj} is cleaned in a SinkBasin and heated before slicing',
                                                       '{obj} is sliced before cleaning in a SinkBasin and heating',
                                                       '{obj} is cooled in a Fridge and heated before slicing',
                                                       '{obj} is sliced before cooling in a Fridge and heating',
                                                       '{obj} is cleaned in a SinkBasin, cooled in a {Fridge}, and '
                                                       'then heated']

gdict['clean_and_heat_then_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['clean_and_heat_then_place']['templates_pos'] = ['{obj} is cleaned in a SinkBasin and heated,'
                                                       ' then placed in a {recep}',
                                                       '{obj} is heated and cleaned in a SinkBasin,'
                                                       ' then placed in a {recep}',
                                                       '{obj} is placed in a {recep} after '
                                                       'cleaning in a SinkBasin and heating',
                                                       '{obj} is placed in a {recep} after heating and cleaning '
                                                       'in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin and heated before '
                                                       'placing in a {recep}',
                                                       '{obj} is heated and cleaned in a SinkBasin before placing in '
                                                       'a {recep}']
gdict['clean_and_heat_then_place']['templates_neg'] = ['{obj} is cleaned in a SinkBasin and cooled in a Fridge '
                                                       'before placing in a {recep}',
                                                       '{obj} is cooled in a Fridge and cleaned in SinkBasin before '
                                                       'placing in a {recep}',
                                                       '{obj} is cleaned in a SinkBasin and heated '
                                                       'before slicing',
                                                       'a clean, sliced {obj} is placed in a {recep}',
                                                       'a cold, clean {obj} is placed in a {recep}',
                                                       'a hot, clean {obj} is sliced in a {recep}',
                                                       '{obj} is placed in a {recep} after cleaning in a SinkBasin '
                                                       'and cooling in a Fridge',
                                                       '{obj} is placed in a {recep} after slicing '
                                                       'and heating']

gdict['clean_and_heat_then_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_and_heat_then_slice']['templates_pos'] = ['{obj} is heated and cleaned in a SinkBasin,'
                                                       ' then sliced',
                                                       '{obj} is cleaned in a SinkBasin and heated,'
                                                       ' then sliced',
                                                       '{obj} is sliced after heating and cleaning in a SinkBasin',
                                                       '{obj} is sliced after cleaning in a SinkBasin and heating',
                                                       '{obj} is heated and cleaned in a SinkBasin before slicing',
                                                       '{obj} is cleaned in a SinkBasin and heated before slicing']
gdict['clean_and_heat_then_slice']['templates_neg'] = ['{obj} is sliced before cleaning in a SinkBasin and heating',
                                                       '{obj} is cleaned in a SinkBasin after heating and slicing',
                                                       '{obj} is heated and cleaned in a SinkBasin after slicing',
                                                       '{obj} is heated and sliced before cleaning in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin and cooled in a Fridge, '
                                                       'then sliced',
                                                       '{obj} is sliced, then cleaned in a SinkBasin and heated']

gdict['clean_and_slice_then_cool']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_and_slice_then_cool']['templates_pos'] = ['{obj} is cleaned in a SinkBasin and sliced,'
                                                       ' then cooled in a Fridge',
                                                       '{obj} is sliced and cleaned in a SinkBasin,'
                                                       ' then cooled in a Fridge',
                                                       '{obj} is cooled in a Fridge after '
                                                       'cleaning in a SinkBasin and slicing',
                                                       '{obj} is cooled in a Fridge after '
                                                       'slicing and cleaning in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin and sliced before '
                                                       'cooling in a Fridge',
                                                       '{obj} is sliced and cleaned in a SinkBasin before '
                                                       'cooling in a Fridge']
gdict['clean_and_slice_then_cool']['templates_neg'] = ['{obj} is cooled in a Fridge, then cleaned in a SinkBasin '
                                                       'and sliced',
                                                       '{obj} is cooled in a Fridge before cleaning in a SinkBasin '
                                                       'and slicing',
                                                       '{obj} is cleaned in a SinkBasin after cooling in a Fridge '
                                                       'and slicing',
                                                       '{obj} is cooled in a Fridge and sliced before cleaning in '
                                                       'a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin and sliced after cooling '
                                                       'in a Fridge',
                                                       '{obj} is cleaned in a SinkBasin and sliced, then heated']

gdict['clean_and_slice_then_heat']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_and_slice_then_heat']['templates_pos'] = ['{obj} is cleaned in a SinkBasin and sliced,'
                                                       ' then heated',
                                                       '{obj} is sliced and cleaned in a SinkBasin,'
                                                       ' then heated',
                                                       '{obj} is heated after cleaning in a SinkBasin and slicing',
                                                       '{obj} is heated after slicing and cleaning in a SinkBasin',
                                                       '{obj} cleaned in a SinkBasin and sliced before heating',
                                                       '{obj} is sliced and cleaned in a SinkBasin before heating']
gdict['clean_and_slice_then_heat']['templates_neg'] = ['{obj} is heated, then cleaned in a SinkBasin '
                                                       'and sliced',
                                                       '{obj} is heated before cleaning in a SinkBasin '
                                                       'and slicing',
                                                       '{obj} is cleaned in a SinkBasin after heating '
                                                       'and slicing',
                                                       '{obj} is heated and sliced before cleaning in '
                                                       'a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin and sliced after heating',
                                                       '{obj} is cleaned in a SinkBasin and sliced, then '
                                                       'cooled in a Fridge']

# cool_and_clean_then_place, cool_and_clean_then_slice, heat_and_clean_then_place,
# heat_and_clean_then_slice, slice_and_clean_then_cool, slice_and_clean_then_heat, slice_and_cool_then_clean
# clean_then_slice_and_cool, clean_then_slice_and_heat, cool_then_slice_and_clean, heat_then_slice_and_clean
# slice_then_cool_and_clean, slice_then_heat_and_clean, slice_and_heat_then_clean
gdict['cool_and_slice_then_clean']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['cool_and_slice_then_clean']['templates_pos'] = ['{obj} is cooled in a Fridge and sliced,'
                                                       ' then cleaned in a SinkBasin',
                                                       '{obj} is sliced and cooled in a Fridge, '
                                                       'then cleaned in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin after '
                                                       'cooling in a Fridge and slicing',
                                                       '{obj} is cleaned in a SinkBasin after '
                                                       'slicing and cooling in a Fridge',
                                                       '{obj} is cooled in Fridge and sliced before '
                                                       'cleaning in SinkBasin',
                                                       '{obj} is sliced and cooled in Fridge before '
                                                       'cleaning in SinkBasin']
gdict['cool_and_slice_then_clean']['templates_neg'] = ['{obj} is cleaned in a SinkBasin before cooling in a Fridge '
                                                       'and slicing',
                                                       '{obj} is cleaned in a SinkBasin and sliced before cooling '
                                                       'in a Fridge',
                                                       '{obj} is cooled in a Fridge after cleaning in a '
                                                       'SinkBasin and slicing',
                                                       '{obj} is cooled in a Fridge and sliced after cleaning in a '
                                                       'SinkBasin',
                                                       '{obj} is heated and sliced, then cleaned in a SinkBasin',
                                                       '{obj} is cooled in a Fridge and cleaned in a SinkBasin '
                                                       'before slicing']

gdict['heat_and_slice_then_clean']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['heat_and_slice_then_clean']['templates_pos'] = ['{obj} is sliced and heated,'
                                                       ' then cleaned in a SinkBasin',
                                                       '{obj} is heated and sliced, then cleaned in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin after slicing '
                                                       'and heating',
                                                       '{obj} is cleaned in a SinkBasin after heating and slicing',
                                                       '{obj} is heated and sliced before '
                                                       'cleaning in SinkBasin',
                                                       '{obj} is sliced and heated before '
                                                       'cleaning in SinkBasin']
gdict['heat_and_slice_then_clean']['templates_neg'] = ['{obj} is cleaned in a SinkBasin before heating and '
                                                       'slicing',
                                                       '{obj} is cleaned in a SinkBasin and sliced before heating',
                                                       '{obj} is heated after cleaning in a '
                                                       'SinkBasin and slicing',
                                                       '{obj} is heated and sliced after cleaning in a '
                                                       'SinkBasin',
                                                       '{obj} is cooled in a Fridge and sliced, then '
                                                       'cleaned in a SinkBasin',
                                                       '{obj} is heated and cleaned in a SinkBasin '
                                                       'before slicing']

gdict['slice_and_clean_then_place']['pddl'] = gdict['slice_and_clean_and_place']['pddl']
gdict['slice_and_clean_then_place']['templates_pos'] = ['{obj} is cleaned in a SinkBasin and sliced,'
                                                        ' then placed in a {recep}',
                                                        '{obj} is sliced and cleaned in a SinkBasin, '
                                                        'then placed in a {recep}',
                                                        '{obj} is placed in a {recep} after '
                                                        'cleaning in a SinkBasin and slicing',
                                                        '{obj} is placed in a {recep} after slicing and cleaning in '
                                                        'a SinkBasin',
                                                        '{obj} is cleaned in a SinkBasin and sliced before '
                                                        'placing in a {recep}',
                                                        '{obj} is sliced and cleaned in a SinkBasin before '
                                                        'placing in a {recep}']
gdict['slice_and_clean_then_place']['templates_neg'] = ['{obj} is cleaned in a SinkBasin and cooled in a Fridge '
                                                        'before placing in a {recep}',
                                                        '{obj} is cooled in a Fridge and cleaned in SinkBasin before '
                                                        'placing in a {recep}',
                                                        '{obj} is cleaned in a SinkBasin and heated '
                                                        'before slicing',
                                                        'a hot, sliced {obj} is placed in a {recep}',
                                                        'a cold, clean {obj} is placed in a {recep}',
                                                        'a hot, clean {obj} is sliced in a {recep}',
                                                        '{obj} is placed in a {recep} after cleaning in a SinkBasin '
                                                        'and cooling in a Fridge',
                                                        '{obj} is placed in a {recep} after slicing '
                                                        'and heating']

gdict['slice_and_cool_then_place']['pddl'] = gdict['cool_and_slice_and_place']['pddl']
gdict['slice_and_cool_then_place']['templates_pos'] = ['{obj} is cooled in a Fridge and sliced,'
                                                       ' then placed in a {recep}',
                                                       '{obj} is sliced and cooled in a Fridge,'
                                                       ' then placed in a {recep}',
                                                       '{obj} is placed in a {recep} after '
                                                       'cooling in a Fridge and slicing',
                                                       '{obj} is placed in a {recep} after '
                                                       'slicing and cooling in a Fridge',
                                                       '{obj} is cooled in a Fridge and sliced before '
                                                       'placing in a {recep}',
                                                       '{obj} is sliced and cooled in a Fridge before '
                                                       'placing in a {recep}']
gdict['slice_and_cool_then_place']['templates_neg'] = ['{obj} is cleaned in a SinkBasin and cooled in a Fridge '
                                                       'before placing in a {recep}',
                                                       '{obj} is heated and sliced before '
                                                       'placing in a {recep}',
                                                       '{obj} is cleaned in a SinkBasin and cooled in a Fridge '
                                                       'before slicing',
                                                       'a hot, sliced {obj} is placed in a {recep}',
                                                       'a cold, clean {obj} is placed in a {recep}',
                                                       'a cold, clean {obj} is sliced in a {recep}',
                                                       '{obj} is placed in a {recep} after cleaning in a SinkBasin '
                                                       'and cooling in a Fridge',
                                                       '{obj} is placed in a {recep} after slicing '
                                                       'and heating']

gdict['slice_and_heat_then_place']['pddl'] = gdict['slice_and_heat_and_place']['pddl']
gdict['slice_and_heat_then_place']['templates_pos'] = ['{obj} is sliced and heated, then placed in a {recep}',
                                                       '{obj} is heated and sliced, then placed in a {recep}',
                                                       '{obj} is placed in a {recep} after slicing and heating',
                                                       '{obj} is placed in a {recep} after heating and slicing',
                                                       '{obj} is sliced and heated before placing in a {recep}',
                                                       '{obj} is heated and sliced before placing in a {recep}']
gdict['slice_and_heat_then_place']['templates_neg'] = ['{obj} is cleaned in a SinkBasin and heated '
                                                       'before placing in a {recep}',
                                                       '{obj} is cooled in a Fridge and sliced before '
                                                       'placing in a {recep}',
                                                       '{obj} is cleaned in a SinkBasin and heated '
                                                       'before slicing',
                                                       'a hot, clean {obj} is placed in a {recep}',
                                                       'a cold, sliced {obj} is placed in a {recep}',
                                                       'a hot, sliced {obj} is cleaned in a {recep}',
                                                       '{obj} is placed in a {recep} after cleaning in a SinkBasin '
                                                       'and heating',
                                                       '{obj} is placed in a {recep} after slicing '
                                                       'and cooling in a Fridge']

###########################################################################################
###########################################################################################

gdict['clean_then_cool_and_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_then_cool_and_slice']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then cooled in a Fridge '
                                                       'and sliced',
                                                       '{obj} is cleaned in a SinkBasin, then sliced and cooled '
                                                       'in a Fridge',
                                                       '{obj} is cooled in a Fridge and sliced after '
                                                       'cleaning in a SinkBasin',
                                                       '{obj} is sliced and cooled in a Fridge after cleaning in a '
                                                       'SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin before cooling in a Fridge '
                                                       'and slicing',
                                                       '{obj} is cleaned in a SinkBasin before slicing and cooling '
                                                       'in a Fridge']
gdict['clean_then_cool_and_slice']['templates_neg'] = ['{obj} is cooled in a Fridge and sliced '
                                                       'before cleaning in a SinkBasin',
                                                       '{obj} is sliced and cooled in a Fridge before cleaning '
                                                       'in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin after slicing '
                                                       'and cooling in a Fridge',
                                                       'a hot {obj} is cooled in a Fridge and sliced',
                                                       'a clean {obj} is sliced and heated',
                                                       'a cool, sliced {obj} is cleaned in a SinkBasin',
                                                       'a sliced, cool {obj} is cleaned in a SinkBasin']

gdict['clean_then_heat_and_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_then_heat_and_slice']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then heated and sliced',
                                                       '{obj} is cleaned in a SinkBasin, then sliced and heated',
                                                       '{obj} is heated and sliced after cleaning in a SinkBasin',
                                                       '{obj} is sliced and heated after cleaning in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin before heating and slicing',
                                                       '{obj} is cleaned in a SinkBasin before slicing and heating']
gdict['clean_then_heat_and_slice']['templates_neg'] = ['{obj} is heated and sliced '
                                                       'before cleaning in a SinkBasin',
                                                       '{obj} is sliced and heated before cleaning in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin after slicing '
                                                       'and heating',
                                                       'a cold {obj} is heated and sliced',
                                                       'a clean {obj} is cooled in a Fridge and sliced',
                                                       'a clean {obj} is sliced and cooled in a Fridge',
                                                       'a hot, sliced {obj} is cleaned in a SinkBasin',
                                                       'a sliced, hot {obj} is cleaned in a SinkBasin']

gdict['cool_then_clean_and_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['cool_then_clean_and_slice']['templates_pos'] = ['{obj} is cooled in a Fridge, then cleaned in a SinkBasin '
                                                       'and sliced',
                                                       '{obj} is cooled in a Fridge, then sliced and cleaned in '
                                                       'a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin and sliced after '
                                                       'cooling in a Fridge',
                                                       '{obj} is sliced and cleaned in a SinkBasin after cooling in '
                                                       'a Fridge',
                                                       '{obj} is cooled in a Fridge before cleaning in a SinkBasin '
                                                       'and slicing',
                                                       '{obj} is cooled in a Fridge before slicing and cleaning in '
                                                       'a SinkBasin']
gdict['cool_then_clean_and_slice']['templates_neg'] = ['{obj} is cleaned in a SinkBasin and sliced '
                                                       'before cooling in a Fridge',
                                                       '{obj} is sliced and cleaned in a SinkBasin before '
                                                       'cooling in a Fridge',
                                                       '{obj} is cooled in a Fridge after cleaning in a SinkBasin '
                                                       'and slicing',
                                                       'a hot {obj} is cleaned in a SinkBasin and sliced',
                                                       'a sliced {obj} is cooled in a Fridge and cleaned '
                                                       'in a SinkBasin',
                                                       'a cold {obj} is cleaned in a SinkBasin and heated',
                                                       'a sliced, clean {obj} is cooled in a Fridge',
                                                       'a clean, sliced {obj} is cooled in a Fridge']

gdict['heat_then_clean_and_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['heat_then_clean_and_slice']['templates_pos'] = ['{obj} is heated, then cleaned in a SinkBasin and sliced',
                                                       '{obj} is heated, then sliced and cleaned in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin and sliced after heating',
                                                       '{obj} is sliced and cleaned in a SinkBasin after heating',
                                                       '{obj} is heated before cleaning in a SinkBasin and slicing',
                                                       '{obj} is heated before slicing and cleaning in a SinkBasin']
gdict['heat_then_clean_and_slice']['templates_neg'] = ['{obj} is cleaned in a SinkBasin and sliced '
                                                       'before heating',
                                                       '{obj} is sliced and cleaned in a SinkBasin before '
                                                       'heating',
                                                       '{obj} is heated after cleaning in a SinkBasin '
                                                       'and slicing',
                                                       'a cold {obj} is cleaned in a SinkBasin and sliced',
                                                       'a sliced {obj} is heated and cleaned in a SinkBasin',
                                                       'a cold {obj} is cleaned in a SinkBasin and sliced',
                                                       'a clean, sliced {obj} is heated',
                                                       'a sliced, clean {obj} is heated']

gdict['slice_then_clean_and_cool']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['slice_then_clean_and_cool']['templates_pos'] = ['{obj} is sliced, then cleaned in a SinkBasin '
                                                       'and cooled in a Fridge',
                                                       '{obj} is sliced, then cooled in a Fridge and cleaned '
                                                       'in a SinkBasin',
                                                       '{obj} is cooled in a Fridge and cleaned in a SinkBasin '
                                                       'after slicing',
                                                       '{obj} is cleaned in a SinkBasin and cooled in a Fridge '
                                                       'after slicing',
                                                       '{obj} is sliced before cleaning in a SinkBasin and'
                                                       ' cooling in a Fridge',
                                                       '{obj} is sliced before cooling in a Fridge and cleaning,'
                                                       ' in a SinkBasin']
gdict['slice_then_clean_and_cool']['templates_neg'] = ['{obj} is cooled in a Fridge and cleaned in a SinkBasin '
                                                       'before slicing',
                                                       '{obj} is sliced after cleaning in a SinkBasin and cooling '
                                                       'in a Fridge',
                                                       'a sliced {obj} is cleaned in a SinkBasin and heated',
                                                       'a cold, clean {obj} is sliced',
                                                       'a clean, cold {obj} is sliced',
                                                       'a clean {obj} is sliced and cooled in a Fridge']

gdict['slice_then_clean_and_heat']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['slice_then_clean_and_heat']['templates_pos'] = ['{obj} is sliced, then cleaned in a SinkBasin and heated',
                                                       '{obj} is sliced, then heated and cleaned in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin and heated '
                                                       'after slicing',
                                                       '{obj} is heated and cleaned in a SinkBasin after slicing',
                                                       '{obj} is sliced before cleaning in a SinkBasin and heating',
                                                       '{obj} is sliced before heating and cleaning in a SinkBasin']
gdict['slice_then_clean_and_heat']['templates_neg'] = ['{obj} is heated and cleaned in a SinkBasin '
                                                       'before slicing',
                                                       '{obj} is cleaned in a SinkBasin and heated before slicing',
                                                       '{obj} is sliced after cleaning in a SinkBasin and heating',
                                                       '{obj} is sliced after heating and cleaning in a SinkBasin',
                                                       'a sliced {obj} is cleaned in a SinkBasin and cooled in a '
                                                       'Fridge',
                                                       'a sliced {obj} is cooled in a Fridge and cleaned in a '
                                                       'SinkBasin',
                                                       'a hot, clean {obj} is sliced',
                                                       'a clean, hot {obj} is sliced',
                                                       'a clean {obj} is sliced and heated']

# clean_then_cool_and_place, clean_then_heat_and_place, clean_then_slice_and_place, slice_then_cool_and_place
# slice_then_heat_and_place, slice_then_clean_and_place, heat_then_clean_and_place, heat_then_slice_and_place
# cool_then_clean_and_place, cool_then_slice_and_place
gdict['clean_then_cool_and_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['clean_then_cool_and_place']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then cooled in a Fridge '
                                                       'and placed in a {recep}',
                                                       '{obj} is cooled in a Fridge and placed in a {recep} after '
                                                       'cleaning in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin before cooling in a Fridge '
                                                       'and placing in a {recep}']
gdict['clean_then_cool_and_place']['templates_neg'] = ['{obj} is cooled in a Fridge and placed in a {recep} '
                                                       'before cleaning in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin after cooling in a Fridge '
                                                       'and placing in a {recep}',
                                                       'a clean {obj} is heated and placed in a {recep}',
                                                       'a clean, hot {obj} is placed in a {recep}',
                                                       'a clean, cold {obj} is sliced']

gdict['clean_then_heat_and_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['clean_then_heat_and_place']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then heated '
                                                       'and placed in a {recep}',
                                                       '{obj} is heated and placed in a {recep} after '
                                                       'cleaning in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin before heating and'
                                                       ' placing in a {recep}']
gdict['clean_then_heat_and_place']['templates_neg'] = ['{obj} is heated and placed in a {recep} '
                                                       'before cleaning in a SinkBasin',
                                                       '{obj} is cleaned in a SinkBasin after heating '
                                                       'and placing in a {recep}',
                                                       'a clean {obj} is cooled and placed in a {recep}',
                                                       'a clean, cold {obj} is placed in a {recep}',
                                                       'a clean, hot {obj} is sliced']

gdict['clean_then_slice_and_place']['pddl'] = gdict['slice_and_clean_and_place']['pddl']
gdict['clean_then_slice_and_place']['templates_pos'] = ['{obj} is cleaned in a SinkBasin, then sliced '
                                                        'and placed in a {recep}',
                                                        '{obj} is sliced and placed in a {recep} after '
                                                        'cleaning in a SinkBasin',
                                                        '{obj} is cleaned in a SinkBasin before slicing and'
                                                        ' placing in a {recep}']
gdict['clean_then_slice_and_place']['templates_neg'] = ['{obj} is sliced and placed in a {recep} '
                                                        'before cleaning in a SinkBasin',
                                                        '{obj} is cleaned in a SinkBasin after slicing '
                                                        'and placing in a {recep}',
                                                        'a clean {obj} is cooled and placed in a {recep}',
                                                        'a cold, sliced {obj} is placed in a {recep}',
                                                        'a clean, sliced {obj} is heated']

gdict['slice_then_cool_and_place']['pddl'] = gdict['cool_and_slice_and_place']['pddl']
gdict['slice_then_cool_and_place']['templates_pos'] = ['{obj} is sliced, then cooled in a Fridge '
                                                       'and placed in a {recep}',
                                                       '{obj} is cooled in a Fridge and placed in a {recep} '
                                                       'after slicing',
                                                       '{obj} is sliced before cooling in a Fridge and'
                                                       ' placing in a {recep}']
gdict['slice_then_cool_and_place']['templates_neg'] = ['{obj} is cooled in a Fridge and placed in a {recep} '
                                                       'before slicing',
                                                       '{obj} is sliced after cooling in a Fridge and '
                                                       'placing in a {recep}',
                                                       'a sliced {obj} is heated and placed in a {recep}',
                                                       'a clean, sliced {obj} is placed in a {recep}',
                                                       'a sliced, cold {obj} is cleaned']

gdict['slice_then_heat_and_place']['pddl'] = gdict['slice_and_heat_and_place']['pddl']
gdict['slice_then_heat_and_place']['templates_pos'] = ['{obj} is sliced, then heated and placed in a {recep}',
                                                       '{obj} is heated and placed in a {recep} after slicing',
                                                       '{obj} is sliced before heating and'
                                                       ' placing in a {recep}']
gdict['slice_then_heat_and_place']['templates_neg'] = ['{obj} is heated and placed in a {recep} '
                                                       'before slicing',
                                                       '{obj} is sliced after heating and '
                                                       'placing in a {recep}',
                                                       'a sliced {obj} is cooled and placed in a {recep}',
                                                       'a clean, hot {obj} is placed in a {recep}',
                                                       'a sliced, hot {obj} is cleaned']

gdict['slice_then_clean_and_place']['pddl'] = gdict['slice_and_clean_and_place']['pddl']
gdict['slice_then_clean_and_place']['templates_pos'] = ['{obj} is sliced, then cleaned in a SinkBasin '
                                                        'and placed in a {recep}',
                                                        '{obj} is cleaned in a SinkBasin and placed in a '
                                                        '{recep} after slicing',
                                                        '{obj} is sliced before cleaning in a SinkBasin and'
                                                        ' placing in a {recep}']
gdict['slice_then_clean_and_place']['templates_neg'] = ['{obj} is cleaned in a SinkBasin and placed in a {recep} '
                                                        'before slicing',
                                                        '{obj} is sliced after cleaning in a SinkBasin and '
                                                        'placing in a {recep}',
                                                        'a sliced {obj} is cooled and placed in a {recep}',
                                                        'a hot, clean {obj} is placed in a {recep}',
                                                        'a sliced, clean {obj} is heated']

gdict['heat_then_clean_and_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['heat_then_clean_and_place']['templates_pos'] = ['{obj} is heated, then cleaned in a SinkBasin and '
                                                       'placed in a {recep}',
                                                       '{obj} is cleaned in a SinkBasin and placed in a {recep} '
                                                       'after heating',
                                                       '{obj} is heated before cleaning in a SinkBasin and'
                                                       ' placing in a {recep}']
gdict['heat_then_clean_and_place']['templates_neg'] = ['{obj} is cleaned in a SinkBasin and placed in a {recep} '
                                                       'before heating',
                                                       '{obj} is heated after cleaning in a SinkBasin and '
                                                       'placing in a {recep}',
                                                       'a cold {obj} is cleaned in a SinkBasin and placed in a {recep}',
                                                       'a hot {obj} is sliced and placed in a {recep}',
                                                       'a cold, clean {obj} is placed in a {recep}',
                                                       'a hot, clean {obj} is sliced']

gdict['heat_then_slice_and_place']['pddl'] = gdict['slice_and_heat_and_place']['pddl']
gdict['heat_then_slice_and_place']['templates_pos'] = ['{obj} is heated, then sliced and placed in a {recep}',
                                                       '{obj} is sliced and placed in a {recep} after heating',
                                                       '{obj} is heated before slicing and placing in a {recep}']
gdict['heat_then_slice_and_place']['templates_neg'] = ['{obj} is sliced and placed in a {recep} before heating',
                                                       '{obj} is heated after slicing and placing in a {recep}',
                                                       'a cold {obj} is sliced and placed in a {recep}',
                                                       'a hot {obj} is cleaned and placed in a {recep}',
                                                       'a cold, sliced {obj} is placed in a {recep}',
                                                       'a heat, sliced {obj} is cleaned']

gdict['cool_then_clean_and_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['cool_then_clean_and_place']['templates_pos'] = ['a cold {obj} is cleaned in a SinkBasin and placed in a {recep}',
                                                       '{obj} is cooled in Fridge, then cleaned in a SinkBasin and '
                                                       'placed in a {recep}',
                                                       '{obj} is cleaned in a SinkBasin and placed in a {recep} '
                                                       'after cooling in a Fridge',
                                                       '{obj} is cooled in a Fridge before cleaning in a SinkBasin '
                                                       'and placing in a {recep}']
gdict['cool_then_clean_and_place']['templates_neg'] = ['{obj} is cleaned in a SinkBasin and placed in a {recep} '
                                                       'before cooling in a Fridge',
                                                       '{obj} is cooled in a Fridge after cleaning in a SinkBasin '
                                                       'and placing in a {recep}',
                                                       'a hot {obj} is cleaned in a SinkBasin and placed in a {recep}',
                                                       'a clean {obj} is cooled in a Fridge and placed in a {recep}',
                                                       'a cold {obj} is cleaned in a SinkBasin and sliced in a {recep}',
                                                       'a cold, clean {obj} is sliced']

gdict['cool_then_slice_and_place']['pddl'] = gdict['cool_and_slice_and_place']['pddl']
gdict['cool_then_slice_and_place']['templates_pos'] = ['{obj} is cooled in a Fridge, then sliced and '
                                                       'placed in a {recep}',
                                                       '{obj} is sliced and placed in a {recep} after '
                                                       'cooling in a Fridge',
                                                       '{obj} is cooled in a Fridge before slicing and '
                                                       'placing in a {recep}']
gdict['cool_then_slice_and_place']['templates_neg'] = ['{obj} is sliced and placed in a {recep} '
                                                       'before cooling in a Fridge',
                                                       '{obj} is cooled in a Fridge after slicing '
                                                       'and placing in a {recep}',
                                                       'a hot {obj} is sliced and placed in a {recep}',
                                                       'a sliced {obj} is cooled in a Fridge and placed in a {recep}',
                                                       'a cold, sliced {obj} is cleaned']
