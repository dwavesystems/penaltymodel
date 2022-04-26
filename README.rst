.. image:: https://img.shields.io/pypi/v/penaltymodel.svg
    :target: https://pypi.python.org/pypi/penaltymodel

.. image:: https://img.shields.io/pypi/pyversions/penaltymodel.svg
    :target: https://pypi.python.org/pypi/penaltymodel

.. image:: https://codecov.io/gh/dwavesystems/penaltymodel/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/penaltymodel

.. image:: https://ci.appveyor.com/api/projects/status/cqfk8il1e4hgg7ih?svg=true
    :target: https://ci.appveyor.com/project/dwave-adtt/penaltymodel

.. image:: https://circleci.com/gh/dwavesystems/penaltymodel.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/penaltymodel

penaltymodel
============

.. index-start-marker

One approach to solve a constraint satisfaction problem (`CSP <https://en.wikipedia.org/wiki/Constraint_satisfaction_problem>`_) using an `Ising model <https://en.wikipedia.org/wiki/Ising_model>`_ or a `QUBO <https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization>`_, is to map each individual constraint in the CSP to a 'small' Ising model or QUBO. This mapping is called a *penalty model*.

Imagine that we want to map an AND clause to a QUBO. In other words, we want the solutions
to the QUBO (the solutions that minimize the energy) to be exactly the valid configurations
of an AND gate. Let ``z = AND(x_1, x_2)``.

Before anything else, let's import that package we will need.

.. code-block:: python

    import penaltymodel
    import dimod
    import networkx as nx

Next, we need to determine the feasible configurations that we wish to target (by making the energy of these configuration in the binary quadratic low).
Below is the truth table representing an AND clause.

.. table:: AND Gate
   :name: tbl_ANDgate

   ====================  ====================  ==================
   ``x_1``               ``x_2``               ``z``
   ====================  ====================  ==================
   0                     0                     0
   0                     1                     0
   1                     0                     0
   1                     1                     1
   ====================  ====================  ==================

The rows of the truth table are exactly the feasible configurations.

.. code-block:: python

    feasible_configurations = [{'x1': 0, 'x2': 0, 'z': 0},
                               {'x1': 1, 'x2': 0, 'z': 0},
                               {'x1': 0, 'x2': 1, 'z': 0},
                               {'x1': 1, 'x2': 1, 'z': 1}]

At this point, we can get a penalty model

.. code-block:: python

    bqm, gap = pm.get_penalty_model(feasible_configurations)

However, if we know the QUBO, we can build the penalty model ourselves. We observe that for the equation:

.. code-block::

    E(x_1, x_2, z) = x_1 x_2 - 2(x_1 + x_2) z + 3 z + 0

We get the following energies for each row in our truth table.

.. image:: https://user-images.githubusercontent.com/8395238/34234533-8da5a364-e5a0-11e7-9d9f-068b4ab3a0fd.png
    :target: https://user-images.githubusercontent.com/8395238/34234533-8da5a364-e5a0-11e7-9d9f-068b4ab3a0fd.png

We can see that the energy is minimized on exactly the desired feasible configurations. So we encode this energy function as a QUBO. We make the offset 0.0 because there is no constant energy offset.

.. code-block:: python

    qubo = dimod.BinaryQuadraticModel({'x1': 0., 'x2': 0., 'z': 3.},
                                   {('x1', 'x2'): 1., ('x1', 'z'): 2., ('x2', 'z'): 2.},
                                   0.0,
                                   dimod.BINARY)

We know from the table that our ground energy is ``0``, but we can calculate it using the qubo to check that this is true for the feasible configuration ``(0, 1, 0)``.

.. code-block:: python

    ground_energy = qubo.energy({'x1': 0, 'x2': 1, 'z': 0})

The last value that we need is the classical gap. This is the difference in energy between the lowest infeasible state and the ground state.

.. image:: https://user-images.githubusercontent.com/8395238/34234545-9c93e5f2-e5a0-11e7-8792-5777a5c4303e.png
    :target: https://user-images.githubusercontent.com/8395238/34234545-9c93e5f2-e5a0-11e7-8792-5777a5c4303e.png

With all of the pieces, we can now build the penalty model.

.. code-block:: python

    classical_gap = 1
    p_model = pm.PenaltyModel.from_specification(spec, qubo, classical_gap, ground_energy)

.. index-end-marker

Installation
------------

.. installation-start-marker

To install the core package:

.. code-block:: bash

    pip install penaltymodel

.. installation-end-marker

License
-------

Released under the Apache License 2.0

Contributing
------------

Ocean's `contributing guide <https://docs.ocean.dwavesys.com/en/stable/contributing.html>`_
has guidelines for contributing to Ocean packages.

Release Notes
~~~~~~~~~~~~~

penaltymodel makes use of `reno <https://docs.openstack.org/reno/>`_ to manage its
release notes.

When making a contribution to penaltymodel that will affect users, create a new
release note file by running

.. code-block:: bash

    reno new your-short-descriptor-here

You can then edit the file created under ``releasenotes/notes/``.
Remove any sections not relevant to your changes.
Commit the file along with your changes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_
for details.
