.. image:: https://travis-ci.org/dwavesystems/penaltymodel_maxgap.svg?branch=master
    :target: https://travis-ci.org/dwavesystems/penaltymodel_maxgap

.. image:: https://readthedocs.org/projects/penaltymodel-cache/badge/?version=latest
    :target: http://penaltymodel-cache.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/dwavesystems/penaltymodel_maxgap/badge.svg?branch=master
    :target: https://coveralls.io/github/dwavesystems/penaltymodel_maxgap?branch=master

.. inclusion-marker-do-not-remove

D-Wave Penalty Model Max-gap
============================

Generates penalty models using smt solvers. Serves as a factory and cache for penaltymodel_.

How it works
------------

On install, penaltymodel_maxgap registers an entry point that can be read by
penaltymodel. It will be used automatically by any project that uses penaltymodel's
:code:`get_penalty_model` function.

Installing
----------

To install:

.. code-block:: bash

    pip install penaltymodel_maxgap

To build from source:

.. code-block:: bash

    git clone https://github.com/dwavesystems/penaltymodel_maxgap.git
    cd penaltymodel_maxgap
    python setup.py install

Note that this library will not function without smt solvers installed.
The solvers are accessed through the pysmt_ package. See the accompanying
pysmt documentation for installing smt solvers.

License
-------

Released under the Apache License 2.0. See LICENSE

.. _penaltymodel: https://github.com/dwavesystems/penaltymodel
.. _pysmt: https://github.com/pysmt/pysmt
