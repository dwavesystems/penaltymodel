.. image:: https://img.shields.io/pypi/v/penaltymodel-maxgap.svg
    :target: https://pypi.python.org/pypi/penaltymodel-maxgap

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
    pip install -e ../penaltymodel_core/
    python setup.py install

Note that this library will not function without smt solvers installed. The solvers
are accessed through the pysmt_ package.

In the standard setup (``pip install`` or ``setup.py install`` above), Z3_ solver is installed
auto-magically. See the accompanying pysmt documentation for installing other smt solvers.

In development mode (``pip install -e`` or ``setup.py develop``) solvers are not installed.
Check pysmt_ documentation to see how to do it manually.

.. _pysmt: https://github.com/pysmt/pysmt
.. _Z3: https://github.com/Z3Prover/z3

.. installation-end-marker

License
-------

Released under the Apache License 2.0. See LICENSE

The bundled Z3_ solver used by pysmt_ is licensed under the MIT license.

