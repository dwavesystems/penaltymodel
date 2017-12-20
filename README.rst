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

This project defines a PenaltyModel class that can be used to represent a penalty model in Python.

Installing
----------

To install:

.. code-block:: bash

    pip install penaltymodel

To build from souce:

.. code-block:: bash
    
    pip install -r requirements.txt
    python setup.py install

License
-------

Released under the Apache License 2.0