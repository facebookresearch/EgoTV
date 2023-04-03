# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


from collections import namedtuple

Predicate = namedtuple("Predicate", "name arity")

def is_variable(term):
    return isinstance(term, int)

def var_string(atom):
    """
    find all variable string (string with first letter capitalized)
    :param atom: atom where variables not replaced with integers
    :return: set of variables string
    """
    variables = set()
    for term in atom.terms:
        if term[0].isupper():
            variables.add(term)
    return variables

def str2atom(s):
    """
    :param s:
    :return: Atom where variables not replaced with integers
    """
    s = s.replace(" ", "")
    left = s.find("(")
    right = s.find(")")
    terms = s[left + 1:right].split(",")
    predicate = Predicate(s[:left], len(terms))
    return Atom(predicate, terms)


class Atom(object):
    def __init__(self, predicate, terms):
        """
        :param predicate: Predicate, the predicate of the atom
        :param terms: tuple of string (or integer) of size 1 or 2.
        use integer 0, 1, 2 as variables
        """
        object.__init__(self)
        self.predicate = predicate
        self.terms = tuple(terms)
        assert len(terms) == predicate.arity

    def __str__(self):
        terms_str = ""
        variable_table = ["X", "Y", "Z", "M", "N"]
        for term in self.terms:
            if isinstance(term, int):
                terms_str += variable_table[term]
            else:
                terms_str += term
            terms_str += ","
        terms_str = terms_str[:-1]
        return self.predicate.name + "(" + terms_str + ")"

    @property
    def arity(self):
        return len(self.terms)

    @property
    def variables(self):
        var = [symbol for symbol in self.terms if isinstance(symbol, str)]
        return var  # list preserves order

    @property
    def constants(self):
        const = [symbol for symbol in self.terms if isinstance(symbol, int)]
        return const  # list preserves order
