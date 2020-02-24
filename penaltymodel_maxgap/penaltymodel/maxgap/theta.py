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
    def __init__(self, *args, **kwargs):
        """Theta is a BQM where the biases are pysmt Symbols.

        Theta is normally constructed using :meth:`.Theta.from_graph`.

        """
        super(Theta, self).__init__(*args, **kwargs)

        # add additional assertions tab
        self.assertions = set()

    @classmethod
    def from_graph(cls, graph, linear_energy_ranges, quadratic_energy_ranges):
        """Create Theta from a graph and energy ranges.

        Args:
            graph (:obj:`networkx.Graph`):
                Provides the structure for Theta.

            linear_energy_ranges (dict):
                A dict of the form {v: (min, max), ...} where min and max are the
                range of values allowed to v.
            quadratic_energy_ranges (dict):
                A dict of the form {(u, v): (min, max), ...} where min and max
                are the range of values allowed to (u, v).

        Returns:
            :obj:`.Theta`

        """
        get_env().enable_infix_notation = True  # not sure why we need this here

        theta = cls.empty(dimod.SPIN)

        theta.add_offset(Symbol('offset', REAL))

        def Linear(v):
            """Create a Symbol for the linear bias including the energy range
            constraints."""
            bias = Symbol('h_{}'.format(v), REAL)

            min_, max_ = linear_energy_ranges[v]

            theta.assertions.add(LE(bias, limitReal(max_)))
            theta.assertions.add(GE(bias, limitReal(min_)))

            return bias

        def Quadratic(u, v):
            """Create a Symbol for the quadratic bias including the energy range
            constraints."""
            bias = Symbol('J_{},{}'.format(u, v), REAL)

            if (v, u) in quadratic_energy_ranges:
                min_, max_ = quadratic_energy_ranges[(v, u)]
            else:
                min_, max_ = quadratic_energy_ranges[(u, v)]

            theta.assertions.add(LE(bias, limitReal(max_)))
            theta.assertions.add(GE(bias, limitReal(min_)))

            return bias

        for v in graph.nodes:
            theta.add_variable(v, Linear(v))

        for u, v in graph.edges:
            theta.add_interaction(u, v, Quadratic(u, v))

        return theta

    def to_bqm(self, model):
        """Given a pysmt model, return a bqm.

        Adds the values of the biases as determined by the SMT solver to a bqm.

        Args:
            model: A pysmt model.

        Returns:
            :obj:`dimod.BinaryQuadraticModel`

        """
        linear = ((v, float(model.get_py_value(bias)))
                  for v, bias in self.linear.items())
        quadratic = ((u, v, float(model.get_py_value(bias)))
                     for (u, v), bias in self.quadratic.items())
        offset = float(model.get_py_value(self.offset))

        return dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.SPIN)
