Overview
========

.. image:: https://travis-ci.org/dwavesystems/penaltymodel.svg?branch=master
    :target: https://travis-ci.org/dwavesystems/penaltymodel
    :alt: Travis Status

.. image:: https://coveralls.io/repos/github/dwavesystems/penaltymodel/badge.svg?branch=master
    :target: https://coveralls.io/github/dwavesystems/penaltymodel?branch=master
    :alt: Coverage Report

.. image:: https://readthedocs.org/projects/penaltymodel/badge/?version=latest
    :target: http://penaltymodel.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Installing
----------

$ pip install penaltymodel

penaltymodel
------------

penaltymodel provides utility code for using penalty models. A penalty model is an Ising problem
or QUBO that has ground states corresponding to a set of feasible configurations. In this way
constraint satisfaction problems can be solved using an Binary Quadratic Model Sampler.

This package contains classes to use penalty models in python, as well as utilities that allow
for the creation of factories and caches for penalty models.

This package does not contain any ability to generate or cache penalty models.

License
-------

Released under the Apache License 2.0
