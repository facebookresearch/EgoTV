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
        'templates_pos': ['a {obj} is cleaned and placed'],
        'templates_neg': ['a {obj} is cooled and placed',
                          'a {obj} is heated and placed',
                          'a {obj} is cleaned and heated']
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
        'templates_pos': ['a hot {obj} is placed',
                          'a {obj} is heated and placed'],
        'templates_neg': ['a cold {obj} is placed',
                          'a {obj} is cleaned and placed',
                          'a sliced {obj}']
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
        'templates_pos': ['a cold {obj} is placed',
                          'a {obj} is cooled and placed'],
        'templates_neg': ['a hot {obj} is placed',
                          'a {obj} is cleaned and placed',
                          'a {obj} is cooled and sliced']
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
        'templates_pos': ['a {obj} is sliced and placed',
                          'a sliced {obj} is placed'],
        'templates_neg': ['a cold {obj} is placed',
                          'a clean {obj}',
                          'a {obj} is heated and placed']
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
        'templates_pos': ['a cold, sliced {obj}',
                          'a sliced, cold {obj}'],
        'templates_neg': ['a cold {obj} is placed',
                          'a hot, sliced {obj}',
                          'a {obj} is sliced and cleaned']
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
        'templates_pos': ['a hot, sliced {obj}',
                          'a sliced, hot {obj}'],
        'templates_neg': ['a cold, sliced {obj}',
                          'a sliced {obj} is placed',
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
        'templates_pos': ['a clean, sliced {obj}',
                          'a sliced, clean {obj}'],
        'templates_neg': ['a hot, clean {obj}',
                          'a sliced, cold {obj}',
                          'a hot, clean, sliced {obj}']
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
        'templates_pos': ['a cold, clean {obj}',
                          'a clean, cold {obj}'],
        'templates_neg': ['a cold, clean, sliced {obj}',
                          'a clean, sliced {obj}',
                          'a hot, clean {obj}']
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
        'templates_pos': ['a hot, clean {obj}',
                          'a clean, hot {obj}'],
        'templates_neg': ['a cold {obj}',
                          'a hot, cleaned, sliced {obj}',
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
        'templates_pos': ['a {obj} in a {mrecep} in a {recep}',
                          'a {obj} in a {mrecep} placed in a {recep}',
                          'a {mrecep} containing a {obj} in a {recep}',
                          'a {mrecep} containing a {obj} placed in a {recep}']
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
        'templates_pos': ['a clean, sliced {obj} is placed',
                          'a sliced, clean {obj} is placed'],
        'templates_neg': ['a cold, sliced {obj} is placed',
                          'a sliced, hot {obj} is placed',
                          'a {obj} is cleaned, sliced and heated',
                          'a clean {obj} is sliced and heated',
                          'a hot {obj} is sliced and cleaned']

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
        'templates_pos': ['a hot, clean, sliced {obj}',
                          'a sliced, hot, clean {obj}'],
        'templates_neg': ['a cold, clean, sliced {obj}',
                          'a sliced, cold, clean {obj}',
                          'a sliced {obj} is cooled',
                          'a clean {obj} is sliced and cooled']
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
        'templates_pos': ['a clean, hot {obj} is placed',
                          'a hot, clean {obj} is placed'],
        'templates_neg': ['a cold, clean {obj} is placed',
                          'a sliced, clean {obj} is placed',
                          'a hot, clean {obj} is sliced',
                          'a clean {obj} is heated and cooled',
                          'a hot {obj} is sliced']
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
        'templates_pos': ['a cold, clean, sliced {obj}',
                          'a sliced, cold, clean {obj}'],
        'templates_neg': ['a hot, clean, sliced {obj}',
                          'a sliced {obj} is cleaned and heated',
                          'a sliced, clean, hot {obj}',
                          'a {obj} is heated and sliced',
                          'a {obj} is heated and cleaned']
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
        'templates_pos': ['a clean, cold {obj} is placed',
                          'a cold, clean {obj} is placed'],
        'templates_neg': ['a hot, clean {obj} is placed',
                          'a hot {obj} is cleaned and cooled',
                          'a clean {obj} is cooled and sliced',
                          'a sliced {obj} is cleaned and cooled',
                          'a clean, sliced {obj} is placed',
                          'a cold, clean {obj} is sliced']
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
        'templates_pos': ['a hot, sliced {obj} is placed',
                          'a sliced, hot {obj} is placed'],
        'templates_neg': ['a cold, sliced {obj} is cleaned',
                          'a hot {obj} is sliced and cleaned',
                          'a clean {obj} is picked and heated',
                          'a hot, sliced, clean {obj}']
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
        'templates_pos': ['a cold, sliced {obj} is placed',
                          'a sliced, cold {obj} is placed'],
        'templates_neg': ['a hot, sliced {obj} is placed',
                          'a clean {obj} is cooled and sliced',
                          'a cold, clean, sliced {obj}',
                          'a {obj} is cleaned and placed',
                          'a cold, clean, sliced {obj} is placed']
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
        'templates_pos': ['a clean {obj} is sliced before placing',
                          'a clean {obj} is placed after slicing'],
        'templates_neg': ['']

    }

# pick, clean, then heat, then place object
gdict["clean_then_heat_then_place"] = \
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
                                    (heatable ?o)
                                    (isHot ?o)
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
        'templates_pos': ['a clean {obj} is heated before placing',
                          'a clean {obj} is placed after heating'],
        'templates_neg': ['']

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
        'templates_pos': ['a hot {obj} is sliced before placing',
                          'a hot {obj} is placed after slicing']
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
        'templates_pos': ['a cold {obj} is sliced before placing',
                          'a cold {obj} is placed after slicing']
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
gdict['clean_then_cool']['templates_pos'] = ['a {obj} is cleaned and then cooled',
                                             'a {obj} is cleaned before cooling',
                                             'a {obj} is cooled after cleaning']
gdict['cool_then_clean']['pddl'] = gdict['cool_and_clean']['pddl']
gdict['cool_then_clean']['templates_pos'] = ['a {obj} is cooled and then cleaned',
                                             'a {obj} is cooled before cleaning',
                                             'a {obj} is cleaned after cooling']

gdict['clean_then_heat']['pddl'] = gdict['clean_and_heat']['pddl']
gdict['clean_then_heat']['templates_pos'] = ['a {obj} is cleaned and then heated',
                                             'a {obj} is cleaned before heating',
                                             'a {obj} is heated after cleaning']
gdict['heat_then_clean']['pddl'] = gdict['clean_and_heat']['pddl']
gdict['heat_then_clean']['templates_pos'] = ['a {obj} is heated and then cleaned',
                                             'a {obj} is heated before cleaning']

gdict['cool_then_slice']['pddl'] = gdict['cool_and_slice']['pddl']
gdict['cool_then_slice']['templates_pos'] = ['a {obj} is cooled and then sliced',
                                             'a {obj} is cooled before slicing',
                                             'a {obj} is sliced after cooling']
gdict['slice_then_cool']['pddl'] = gdict['cool_and_slice']['pddl']
gdict['slice_then_cool']['templates_pos'] = ['a {obj} is sliced and then cooled',
                                             'a {obj} is sliced before cooling',
                                             'a {obj} is cooled after slicing']

gdict['clean_then_slice']['pddl'] = gdict['clean_and_slice']['pddl']
gdict['clean_then_slice']['templates_pos'] = ['a {obj} is cleaned and then sliced',
                                              'a {obj} is cleaned before slicing',
                                              'a {obj} is sliced after cleaning']
gdict['slice_then_clean']['pddl'] = gdict['clean_and_slice']['pddl']
gdict['slice_then_clean']['templates_pos'] = ['a {obj} is sliced and then cleaned',
                                              'a {obj} is sliced before cleaning',
                                              'a {obj} is cleaned after slicing']

gdict['heat_then_slice']['pddl'] = gdict['heat_and_slice']['pddl']
gdict['heat_then_slice']['templates_pos'] = ['a {obj} is heated and then sliced',
                                             'a {obj} is heated before slicing',
                                             'a {obj} is sliced after heating']
gdict['slice_then_heat']['pddl'] = gdict['heat_and_slice']['pddl']
gdict['slice_then_heat']['templates_pos'] = ['a {obj} is sliced and then heated',
                                             'a {obj} is sliced before heating',
                                             'a {obj} is heated after slicing']

gdict['slice_then_place']['pddl'] = gdict['slice_and_place']['pddl']
gdict['slice_then_place']['templates_pos'] = ['a {obj} is sliced and then placed',
                                              'a {obj} is sliced before placing',
                                              'a {obj} is placed after slicing']
gdict['clean_then_place']['pddl'] = gdict['clean_and_place']['pddl']
gdict['clean_then_place']['templates_pos'] = ['a {obj} is cleaned and then placed',
                                              'a {obj} is cleaned before placing',
                                              'a {obj} is placed after cleaning']
gdict['cool_then_place']['pddl'] = gdict['cool_and_place']['pddl']
gdict['cool_then_place']['templates_pos'] = ['a {obj} is cooled and then placed',
                                             'a {obj} is cooled before placing',
                                             'a {obj} is placed after cooling']
gdict['heat_then_place']['pddl'] = gdict['heat_and_place']['pddl']
gdict['heat_then_place']['templates_pos'] = ['a {obj} is heated and then placed',
                                             'a {obj} is heated before placing',
                                             'a {obj} is placed after heating']

# ['cool_and_slice_and_place', 'cool_and_clean_and_place', 'cool_and_slice_and_clean', 'slice_and_heat_and_clean',
# 'slice_and_heat_and_place', 'slice_and_clean_and_place', 'heat_and_clean_and_place']
###########################################################################################
###########################################################################################

gdict['clean_then_cool_then_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['clean_then_cool_then_place']['templates_pos'] = ['a {obj} is cleaned, then cooled, and then placed']

gdict['clean_then_cool_then_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_then_cool_then_slice']['templates_pos'] = ['a {obj} is cleaned, then cooled, and then sliced']

gdict['clean_then_heat_then_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['clean_then_heat_then_place']['templates_pos'] = ['a {obj} is cleaned, then heated, and then placed']

gdict['clean_then_heat_then_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_then_heat_then_slice']['templates_pos'] = ['a {obj} is cleaned, then heated, and then sliced']

gdict['clean_then_slice_then_cool']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_then_slice_then_cool']['templates_pos'] = ['a {obj} is cleaned, then sliced, and then cooled']

gdict['clean_then_slice_then_heat']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_then_slice_then_heat']['templates_pos'] = ['a {obj} is cleaned, then sliced, and then heated']

gdict['cool_then_clean_then_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['cool_then_clean_then_place']['templates_pos'] = ['a {obj} is cooled, then cleaned, and then placed']

gdict['cool_then_clean_then_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['cool_then_clean_then_slice']['templates_pos'] = ['a {obj} is cooled, then cleaned, and then sliced']

gdict['cool_then_slice_then_clean']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['cool_then_slice_then_clean']['templates_pos'] = ['a {obj} is cooled, then sliced, and then cleaned']

gdict['heat_then_clean_then_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['heat_then_clean_then_place']['templates_pos'] = ['a {obj} is heated, then cleaned, and then placed']

gdict['heat_then_clean_then_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['heat_then_clean_then_slice']['templates_pos'] = ['a {obj} is heated, then cleaned, and then sliced']

gdict['heat_then_slice_then_clean']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['heat_then_slice_then_clean']['templates_pos'] = ['a {obj} is heated, then sliced, and then cleaned']

gdict['slice_then_clean_then_cool']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['slice_then_clean_then_cool']['templates_pos'] = ['a {obj} is sliced, then cleaned, and then cooled']

gdict['slice_then_clean_then_heat']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['slice_then_clean_then_heat']['templates_pos'] = ['a {obj} is sliced, then cleaned, and then heated']

gdict['slice_then_clean_then_place']['pddl'] = gdict['slice_and_clean_and_place']['pddl']
gdict['slice_then_clean_then_place']['templates_pos'] = ['a {obj} is sliced, then cleaned, and then placed']

gdict['slice_then_cool_then_clean']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['slice_then_cool_then_clean']['templates_pos'] = ['a {obj} is sliced, then cooled, and then cleaned']

gdict['slice_then_cool_then_place']['pddl'] = gdict['cool_and_slice_and_place']['pddl']
gdict['slice_then_cool_then_place']['templates_pos'] = ['a {obj} is sliced, then cooled, and then placed']

gdict['slice_then_heat_then_clean']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['slice_then_heat_then_clean']['templates_pos'] = ['a {obj} is sliced, then heated, and then cleaned']

gdict['slice_then_heat_then_place']['pddl'] = gdict['slice_and_heat_and_place']['pddl']
gdict['slice_then_heat_then_place']['templates_pos'] = ['a {obj} is sliced, then heated, and then placed']

###########################################################################################
###########################################################################################

gdict['clean_and_cool_then_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['clean_and_cool_then_place']['templates_pos'] = ['a {obj} is cleaned and cooled, and then placed',
                                                       'a {obj} is placed after cleaning and cooling',
                                                       'a {obj} is cleaned and cooled before placing']

gdict['clean_and_cool_then_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_and_cool_then_slice']['templates_pos'] = ['a {obj} is cleaned and cooled, and then sliced',
                                                       'a {obj} is sliced after cleaning and cooling',
                                                       'a {obj} is cleaned and cooled before sliced']

gdict['clean_and_heat_then_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['clean_and_heat_then_place']['templates_pos'] = ['a {obj} is cleaned and heated, and then placed',
                                                       'a {obj} is placed after cleaning and heating',
                                                       'a {obj} is cleaned and heated before placing']

gdict['clean_and_heat_then_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_and_heat_then_slice']['templates_pos'] = ['a {obj} is cleaned and heated, and then sliced',
                                                       'a {obj} is sliced after cleaning and heating',
                                                       'a {obj} is cleaned and heated before slicing']

gdict['clean_and_slice_then_cool']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_and_slice_then_cool']['templates_pos'] = ['a {obj} is cleaned and sliced, and then cooled',
                                                       'a {obj} is cooled after cleaning and slicing',
                                                       'a {obj} is cleaned and sliced before cooling']

gdict['clean_and_slice_then_heat']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_and_slice_then_heat']['templates_pos'] = ['a {obj} is cleaned and sliced, and then heated',
                                                       'a {obj} is heated after cleaning and slicing',
                                                       'a {obj} is cleaned and sliced before heating']

# cool_and_clean_then_place, cool_and_clean_then_slice, heat_and_clean_then_place,
# heat_and_clean_then_slice, slice_and_clean_then_cool, slice_and_clean_then_heat, slice_and_cool_then_clean
# clean_then_slice_and_cool, clean_then_slice_and_heat, cool_then_slice_and_clean, heat_then_slice_and_clean
# slice_then_cool_and_clean, slice_then_heat_and_clean, slice_and_heat_then_clean
gdict['cool_and_slice_then_clean']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['cool_and_slice_then_clean']['templates_pos'] = ['a {obj} is cooled and sliced, and then cleaned',
                                                       'a {obj} is cleaned after cooling and slicing',
                                                       'a {obj} is cooled and sliced before cleaning']

gdict['heat_and_slice_then_clean']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['heat_and_slice_then_clean']['templates_pos'] = ['a {obj} is heated and sliced, and then cleaned',
                                                       'a {obj} is cleaned after heating and slicing',
                                                       'a {obj} is heated and sliced before cleaning']

gdict['slice_and_clean_then_place']['pddl'] = gdict['slice_and_clean_and_place']['pddl']
gdict['slice_and_clean_then_place']['templates_pos'] = ['a {obj} is cleaned and sliced, and then placed',
                                                        'a {obj} is placed after cleaning and slicing',
                                                        'a {obj} is cleaned and sliced before placing']

gdict['slice_and_cool_then_place']['pddl'] = gdict['cool_and_slice_and_place']['pddl']
gdict['slice_and_clean_then_place']['templates_pos'] = ['a {obj} is cooled and sliced, and then placed',
                                                        'a {obj} is placed after cooling and slicing',
                                                        'a {obj} is cooled and sliced before placing']

gdict['slice_and_heat_then_place']['pddl'] = gdict['slice_and_heat_and_place']['pddl']
gdict['slice_and_heat_then_place']['templates_pos'] = ['a {obj} is sliced and heated, and then placed',
                                                       'a {obj} is placed after slicing and heating',
                                                       'a {obj} is sliced and heated before placing']

###########################################################################################
###########################################################################################

gdict['clean_then_cool_and_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['clean_then_cool_and_slice']['templates_pos'] = ['a {obj} is cleaned, then cooled and sliced',
                                                       'a {obj} is cooled and sliced after cleaning',
                                                       'a {obj} is cleaned before cooling and slicing']

gdict['clean_then_heat_and_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['clean_then_heat_and_slice']['templates_pos'] = ['a {obj} is cleaned, then heated and sliced',
                                                       'a {obj} is heated and sliced after cleaning',
                                                       'a {obj} is cleaned before heating and slicing']

gdict['cool_then_clean_and_slice']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['cool_then_clean_and_slice']['templates_pos'] = ['a {obj} is cooled, then cleaned and sliced',
                                                       'a {obj} is cleaned and sliced after cooling',
                                                       'a {obj} is cooled before cleaning and slicing']

gdict['heat_then_clean_and_slice']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['heat_then_clean_and_slice']['templates_pos'] = ['a {obj} is heated, then cleaned and sliced',
                                                       'a {obj} is cleaned and sliced after heating',
                                                       'a {obj} is heated before cleaning and slicing']

gdict['slice_then_clean_and_cool']['pddl'] = gdict['cool_and_slice_and_clean']['pddl']
gdict['slice_then_clean_and_cool']['templates_pos'] = ['a {obj} is sliced, then cleaned and cooled',
                                                       'a {obj} is cleaned and cooled after slicing',
                                                       'a {obj} is sliced before cleaning and cooling']

gdict['slice_then_clean_and_heat']['pddl'] = gdict['slice_and_heat_and_clean']['pddl']
gdict['slice_then_clean_and_heat']['templates_pos'] = ['a {obj} is sliced, then cleaned and heated',
                                                       'a {obj} is cleaned and heated after slicing',
                                                       'a {obj} is sliced before cleaning and heating']

# clean_then_cool_and_place, clean_then_heat_and_place, clean_then_slice_and_place, slice_then_cool_and_place
# slice_then_heat_and_place, slice_then_clean_and_place, heat_then_clean_and_place, heat_then_slice_and_place
# cool_then_clean_and_place, cool_then_slice_and_place
gdict['clean_then_cool_and_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['clean_then_cool_and_place']['templates_pos'] = ['a {obj} is cleaned, then cooled and placed',
                                                       'a {obj} is cooled and placed after cleaning',
                                                       'a {obj} is cleaned before cooling and placing']

gdict['clean_then_heat_and_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['clean_then_heat_and_place']['templates_pos'] = ['a {obj} is cleaned, then heated and placed',
                                                       'a {obj} is heated and placed after cleaning',
                                                       'a {obj} is cleaned before heating and placing']

gdict['clean_then_slice_and_place']['pddl'] = gdict['slice_and_clean_and_place']['pddl']
gdict['clean_then_slice_and_place']['templates_pos'] = ['a {obj} is cleaned, then sliced and placed',
                                                        'a {obj} is sliced and placed after cleaning',
                                                        'a {obj} is cleaned before slicing and placing']

gdict['slice_then_cool_and_place']['pddl'] = gdict['cool_and_slice_and_place']['pddl']
gdict['slice_then_cool_and_place']['templates_pos'] = ['a {obj} is sliced, then cooled and placed',
                                                       'a {obj} is cooled and placed after slicing',
                                                       'a {obj} is sliced before cooling and placing']

gdict['slice_then_heat_and_place']['pddl'] = gdict['slice_and_heat_and_place']['pddl']
gdict['slice_then_heat_and_place']['templates_pos'] = ['a {obj} is sliced, then heated and placed',
                                                       'a {obj} is heated and placed after slicing',
                                                       'a {obj} is sliced before heating and placing']

gdict['slice_then_clean_and_place']['pddl'] = gdict['slice_and_clean_and_place']['pddl']
gdict['slice_then_cool_and_place']['templates_pos'] = ['a {obj} is sliced, then cleaned and placed',
                                                       'a {obj} is cleaned and placed after slicing',
                                                       'a {obj} is sliced before cleaning and placing']

gdict['heat_then_clean_and_place']['pddl'] = gdict['heat_and_clean_and_place']['pddl']
gdict['heat_then_clean_and_place']['templates_pos'] = ['a {obj} is heated, then cleaned and placed',
                                                       'a {obj} is cleaned and placed after heating',
                                                       'a {obj} is heated before cleaning and placing']

gdict['heat_then_slice_and_place']['pddl'] = gdict['slice_and_heat_and_place']['pddl']
gdict['heat_then_slice_and_place']['templates_pos'] = ['a {obj} is heated, then sliced and placed',
                                                       'a {obj} is sliced and placed after heating',
                                                       'a {obj} is heated before slicing and placing']

gdict['cool_then_clean_and_place']['pddl'] = gdict['cool_and_clean_and_place']['pddl']
gdict['cool_then_clean_and_place']['templates_pos'] = ['a cold {obj} is cleaned and placed'
                                                       'a {obj} is cooled, then cleaned and placed',
                                                       'a {obj} is cleaned and placed after cooling',
                                                       'a {obj} is cooled before cleaning and placing']

gdict['cool_then_slice_and_place']['pddl'] = gdict['cool_and_slice_and_place']['pddl']
gdict['cool_then_slice_and_place']['templates_pos'] = ['a {obj} is cooled, then sliced and placed',
                                                       'a {obj} is sliced and placed after cooling',
                                                       'a {obj} is cooled before slicing and placing']
