#########################################
# Common:
# {obj} - ObjectStr
# {recep} - RecepStr
# usage: .format(obj=constants.OBJECTS[self.object_target], recep=constants.OBJECTS[self.parent_target])

# NOTE: order of and/or conditions matters
#########################################

gdict = {}

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
    'templates': ['cool a {obj}',
                   'fetch and cool a {obj}']
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
    'templates': ['heat a {obj}',
                   'fetch and heat a {obj}']
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
    'templates': ['clean a {obj}',
                   'fetch and clean a {obj}']
}

# basic locate and pick (e.g: "locate a apple")
gdict["locate_simple"] = \
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
    'templates': ['locate a {obj}',
                   'fetch a {obj}']
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
    'templates': ['slice a {obj}',
                   'fetch and slice a {obj}']
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
    'templates': ['look at a {obj} under {toggle}',
                  'examine a {obj} under {toggle}']
}

# basic pick and place (e.g: "put the apple in the microwave")
gdict["place_simple"] = \
{
    'pddl' :
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
    'templates': ['put a {obj} in {recep}',
                   'fetch and place a {obj} in {recep}']
}

####################################################################################
# LEVEL 2: composition of basic skills and interactions with objects + quantifiers
####################################################################################

# pick, clean (in sink), place object
gdict["clean_and_place"] = \
{
    'pddl' :
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
    'templates': ['provide a clean {obj} in {recep}']
}


# pick, heat (in microwave), place object
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
    'templates': ['provide a hot {obj} in {recep}']
}


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
    'templates': ['provide a cool {obj} in {recep}']
}

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
    'templates': ['provide a sliced {obj} in {recep}']
}

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
    'templates': ['cool and slice a {obj}']
}


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
        'templates': ['provide two {obj}s in {recep}']
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
        'templates': ['provide a {mrecep} of {obj} in {recep}',
                      'provide {obj} {mrecep} in {recep}']
    }


##########################################################################
# LEVEL 3: complex composition of basic skills and interactions with objects
##########################################################################

# pick, heat, place with movable receptacle
gdict["heat_stack_and_place"] = \
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
        'templates': ['provide a hot {obj} {mrecep} in {recep}']
    }

# pick, cool, place with movable receptacle
gdict["cool_stack_and_place"] = \
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
        'templates': ['provide a cool {obj} {mrecep} in {recep}']
    }

# pick, clean, place with movable receptacle
gdict["clean_stack_and_place"] = \
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
        'templates': ['provide a clean {obj} {mrecep} in {recep}']
    }


# slice, place with movable receptacle
gdict["slice_stack_and_place"] = \
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
        'templates': ['provide a sliced {obj} {mrecep} in {recep}']
    }


# pick, clean, slice, and place object
gdict["clean_slice_and_place"] = \
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
        'templates': ['provide a clean sliced {obj} in {recep}']

    }

# pick, heat, slice & place
gdict["heat_slice_and_place"] = \
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
        'templates': ['provide a hot sliced {obj} in {recep}']
    }

# pick, cool, slice & place
gdict["cool_slice_and_place"] = \
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
        'templates': ['provide a cool sliced {obj} in {recep}']
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
        'templates': ['provide two sliced {obj}s in {recep}']
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
        'templates': ['look at sliced {obj} under the {toggle}',
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
    'templates': ['put all {obj}s in {recep}',
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
        'templates': ['put three {obj}s in {recep}',
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
        'templates': ['Provide a hot {mrecep} of {obj} in {recep}']
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
        'templates': ['Provide a cold {mrecep} of {obj} in {recep}']
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
        'templates': ['Provide a cold {mrecep} of {obj} in {recep}']
    }

###########################################################################################################
###########################################################################################################
# Axis 2: partial ordering
# Level 2
###########################################################################################################
############################################################################################################

gdict["clean_then_heat"] = \
{
    'pddl' :
    '''
        (:goal
            (and
                (forall (?re # receptacle)
                    (not (opened ?re))
                )
                (exists (?o # object)
                    (and 
                        (cleanable ?o)
                        (heatable ?o)
                        (objectType ?o {obj}Type) 
                        (isClean ?o)
                        (isHot ?o)
                    )
                )
            )
        )
    )
    ''',
    'templates': ['heat a clean {obj}',
                  'heat a {obj} after cleaning it',
                  'clean a {obj} then heat it']
}

# pick, clean (in sink), then place object
gdict["clean_then_place"] = \
{
    'pddl' :
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
    'templates': ['clean a {obj} and place it in {recep}',
                  'place a {obj} in {recep} after cleaning it',
                  'clean a {obj} then place it in {recep}']
}


# pick, heat (in microwave), then place object
gdict["heat_then_place"] = \
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
    'templates': ['heat a {obj} and place it in {recep}',
                  'place a {obj} in {recep} after heating it',
                  'heat a {obj} then place it in {recep}']
}


# pick, cool (in refrigerator if not already cool), then place object
gdict["cool_then_place"] = \
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
    'templates': ['cool a {obj} and place it in {recep}',
                  'place a {obj} in {recep} after cooling it',
                  'cool a {obj} then place it in {recep}']
}

# slice, place object
gdict["slice_then_place"] = \
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
    'templates': ['slice a {obj} and place it in {recep}',
                  'place a {obj} in {recep} after slicing it',
                  'slice a {obj} then place it in {recep}']
}

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
        'templates': ['slice a clean {obj} and place it in the {recep}',
                      'clean a {obj}, then slice it and place it in {recep}']

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
        'templates': ['heat a clean {obj} and place it in the {recep}',
                      'clean a {obj}, then heat it and place it in {recep}']

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
        'templates': ['slice a hot {obj} and place it in the {recep}',
                      'heat a {obj}, then slice it and place it in {recep}']
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
        'templates': ['slice a cold {obj} and place it in the {recep}',
                      'cool a {obj}, then slice it and place it in {recep}']
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
        'templates': ['slice two {obj}s and put them in {recep}',
                      'put two sliced {obj}s in {recep}']
    }