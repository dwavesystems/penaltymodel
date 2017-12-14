..  -*- coding: utf-8 -*-

.. _contents:

Overview
========

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

Released under the Apache License 2.0 (see License).

Documentation
-------------

.. only:: html

    :Release: |version|
    :Date: |today|

.. toctree::
   :maxdepth: 1

   penaltymodel_code
   using_factories
   license


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :ref:`glossary`