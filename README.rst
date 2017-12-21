D-Wave Penalty Model
====================

.. image:: https://travis-ci.org/dwavesystems/penaltymodel.svg?branch=master
    :target: https://travis-ci.org/dwavesystems/penaltymodel
    :alt: Travis Status

.. image:: https://coveralls.io/repos/github/dwavesystems/penaltymodel/badge.svg?branch=master
    :target: https://coveralls.io/github/dwavesystems/penaltymodel?branch=master
    :alt: Coverage Report

.. image:: https://readthedocs.org/projects/penaltymodel/badge/?version=latest
    :target: http://penaltymodel.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. inclusion-marker-do-not-remove

One approach to solve a constraint satisfaction problem (`CSP <https://en.wikipedia.org/wiki/Constraint_satisfaction_problem>`_) using an `Ising model <https://en.wikipedia.org/wiki/Ising_model>`_ or a `QUBO <https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization>`_, is to map each individual constraint in the CSP to a 'small' Ising model or QUBO. This mapping is called a *penalty model*.

Installation
------------

To install:

.. code-block:: bash

    pip install penaltymodel

To build from souce:

.. code-block:: bash
    
    pip install -r requirements.txt
    python setup.py install

Example Usage
-------------

Imagine that we want to map an AND clause to a QUBO. In other words, we want the solutions
to the QUBO (the solutions that minimize the energy) to be exactly the valid configurations
of an AND gate. Let :math:`z = AND(x_1, x_2)`.

Before anything else, let's import that package we will need.

.. code-block:: python

    import penaltymodel as pm
    import networkx as nx

Next, we need to determine the feasible configurations that we wish to target (by making the energy of these configuration in the binary quadratic low).
Below is the truth table representing an AND clause.

.. table:: AND Gate
   :name: tbl_ANDgate
 
   ====================  ====================  ==================
   :math:`x_1`           :math:`x_2`           :math:`z`
   ====================  ====================  ==================
   0                     0                     0        
   0                     1                     0           
   1                     0                     0           
   1                     1                     1        
   ====================  ====================  ==================

The rows of the truth table are exactly the feasible configurations.

.. code-block:: python

    feasible_configurations = {(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)}

We also need a target graph and to label the decision variables. We create a node in the graph for each variable in the problem, and we add an edge between each node, represnting the interactions between the variables. In this case we allow an interaction between every variable, but more sparse interactions are possible. The labels of the nodes and the decision variables match.

.. code-block:: python

    graph = nx.Graph()
    graph.add_edges_from([('x1', 'x2'), ('x1', 'z'), ('x2', 'z')])
    decision_variables = ['x1', 'x2', 'z']

We now have everything needed to build our Specification. We have binary variables so we select the appropriate variable type.

.. code-block:: python

    spec = pm.Specification(graph, decision_variables, feasible_configurations, pm.BINARY)

At this point, if we have any factories installed, we could use the factory interface to get an appropriate penalty model for our specification.

.. code-block:: python

    p_model = pm.get_penalty_model(spec)

However, if we know the QUBO, we can build the penalty model ourselves. We observe that for the equation:

.. math::

    E(x_1, x_2, z) = x_1 x_2 - 2(x_1 + x_2) z + 3 z + 0

We get the following energies for each row in our truth table.

.. image:: https://user-images.githubusercontent.com/8395238/34234533-8da5a364-e5a0-11e7-9d9f-068b4ab3a0fd.png
    :target: https://user-images.githubusercontent.com/8395238/34234533-8da5a364-e5a0-11e7-9d9f-068b4ab3a0fd.png

We can see that the energy is minimized on exactly the desired feasible configurations. So we encode this energy function as a QUBO. We make the offset 0.0 because there is no constant energy offset.

.. code-block:: python

    qubo = pm.BinaryQuadraticModel({'x1': 0., 'x2': 0., 'z': 3.},
                                   {('x1', 'x2'): 1., ('x1', 'z'): 2., ('x2', 'z'): 2.},
                                   0.0,
                                   pm.BINARY)

We know from the table that our ground energy is :math:`0`, but we can calculate it using the qubo to check that this is true for the feasible configuration :math:`(0, 1, 0)`.

.. code-block:: python

    ground_energy = qubo.energy({'x1': 0, 'x2': 1, 'x3': 0})

The last value that we need is the classical gap. This is the difference in energy between the lowest infeasible state and the ground state.

.. image:: https://user-images.githubusercontent.com/8395238/34234545-9c93e5f2-e5a0-11e7-8792-5777a5c4303e.png
    :target: https://user-images.githubusercontent.com/8395238/34234545-9c93e5f2-e5a0-11e7-8792-5777a5c4303e.png

With all of the pieces, we can now build the penalty model.

.. code-block:: python

    classical_gap = 1
    p_model = pm.PenaltyModel.from_specification(spec, qubo, classical_gap, ground_energy)

License
-------

Released under the Apache License 2.0