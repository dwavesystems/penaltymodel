# Copyright 2019 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================
from fractions import Fraction

import dimod

from pysmt.environment import get_env
from pysmt.shortcuts import Real, Symbol
from pysmt.shortcuts import LE, GE
from pysmt.typing import REAL


def limitReal(x, max_denominator=1000000):
    """Creates an pysmt Real constant from x.

    Args:
        x (number): A number to be cast to a pysmt constant.
        max_denominator (int, optional): The maximum size of the denominator.
            Default 1000000.

    Returns:
        A Real constant with the given value and the denominator limited.

    """
    f = Fraction(x).limit_denominator(max_denominator)
    return Real((f.numerator, f.denominator))


class Theta(dimod.BinaryQuadraticModel):
    def __init__(self, linear, quadratic, offset, vartype):
        dimod.BinaryQuadraticModel.__init__(self, linear, quadratic, offset, vartype)

        # add additional assertions tab
        self.assertions = set()

    @classmethod
    def from_graph(cls, graph, linear_energy_ranges, quadratic_energy_ranges):
        get_env().enable_infix_notation = True  # not sure why we need this here

        theta = cls.empty(dimod.SPIN)

        theta.add_offset(Symbol('offset', REAL))

        def Linear(v):
            bias = Symbol('h_{}'.format(v), REAL)

            min_, max_ = linear_energy_ranges[v]

            theta.assertions.add(LE(bias, limitReal(max_)))
            theta.assertions.add(GE(bias, limitReal(min_)))

            return bias

        for v in graph.nodes:
            theta.add_variable(v, Linear(v))

        def Quadratic(u, v):
            bias = Symbol('J_{},{}'.format(u, v), REAL)

            if (v, u) in quadratic_energy_ranges:
                min_, max_ = quadratic_energy_ranges[(v, u)]
            else:
                min_, max_ = quadratic_energy_ranges[(u, v)]

            theta.assertions.add(LE(bias, limitReal(max_)))
            theta.assertions.add(GE(bias, limitReal(min_)))

            return bias

        for u, v in graph.edges:
            theta.add_interaction(u, v, Quadratic(u, v))

        return theta

    def to_bqm(self, model):
        linear = ((v, float(model.get_py_value(bias)))
                  for v, bias in self.linear.items())
        quadratic = ((u, v, float(model.get_py_value(bias)))
                     for (u, v), bias in self.quadratic.items())
        offset = float(model.get_py_value(self.offset))

        return dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.SPIN)
