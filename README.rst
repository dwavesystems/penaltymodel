:warning: *penaltymodel* is deprecated. For solving problems with constraints,
    we recommend using the hybrid solvers in the Leap :tm: service. You can find
    documentation for the hybrid solvers at https://docs.dwavequantum.com.

.. image:: https://img.shields.io/pypi/v/penaltymodel.svg
    :target: https://pypi.python.org/pypi/penaltymodel

.. image:: https://img.shields.io/pypi/pyversions/penaltymodel.svg
    :target: https://pypi.python.org/pypi/penaltymodel

.. image:: https://codecov.io/gh/dwavesystems/penaltymodel/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/penaltymodel

.. image:: https://circleci.com/gh/dwavesystems/penaltymodel.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/penaltymodel

============
penaltymodel
============

.. start_penaltymodel_about

One approach to solve a constraint satisfaction problem
(`CSP <https://en.wikipedia.org/wiki/Constraint_satisfaction_problem>`_) using
an `Ising model <https://en.wikipedia.org/wiki/Ising_model>`_ or a
`QUBO <https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization>`_,
is to map each individual constraint in the CSP to a 'small' Ising model or
QUBO. This mapping is called a *penalty model*.

.. end_penaltymodel_about

For more information, see
`penalty models <https://docs.dwavequantum.com/en/latest/concepts/penalty.html>`_.

Installation
============

To install the core package:

.. code-block:: bash

    pip install penaltymodel

License
=======

Released under the Apache License 2.0

Contributing
============

Ocean's
`contributing guide <https://docs.dwavequantum.com/en/latest/ocean/contribute.html>`_
has guidelines for contributing to Ocean packages.

Release Notes
-------------

penaltymodel makes use of `reno <https://docs.openstack.org/reno/>`_ to manage
its release notes.

When making a contribution to penaltymodel that will affect users, create a new
release note file by running

.. code-block:: bash

    reno new your-short-descriptor-here

You can then edit the file created under ``releasenotes/notes/``.
Remove any sections not relevant to your changes.
Commit the file along with your changes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_
for details.