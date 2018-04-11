D-Wave Penalty Model Max-gap
============================

.. index-start-marker

Generates penalty models using smt solvers. Serves as a factory and cache for penaltymodel.

On install, penaltymodel_maxgap registers an entry point that can be read by
penaltymodel. It will be used automatically by any project that uses penaltymodel's
:code:`get_penalty_model` function.

.. index-end-marker

Installing
----------

.. installation-start-marker

To install:

.. code-block:: bash

    pip install penaltymodel_maxgap

To build from source:

.. code-block:: bash

    cd penaltymodel_maxgap
    pip install -r requirements.txt
    python setup.py install

Note that this library will not function without smt solvers installed.
The solvers are accessed through the pysmt_ package. See the accompanying
pysmt documentation for installing smt solvers.

.. _pysmt: https://github.com/pysmt/pysmt

.. installation-end-marker

License
-------

Released under the Apache License 2.0. See LICENSE
