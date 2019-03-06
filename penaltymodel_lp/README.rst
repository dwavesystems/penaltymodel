.. image:: https://img.shields.io/pypi/v/penaltymodel-lp.svg
    :target: https://pypi.python.org/pypi/penaltymodel-lp

Penalty Model - Linear Programming
==================================================

.. index-start-marker

Generates penalty models using `scipy.optimize`_'s Linear Programming capability.
Serves as a factory and cache for penaltymodel.

On install, penaltymodel-lp registers an entry point that can be read by
penaltymodel. It will be used automatically by any project that uses penaltymodel's
:code:`get_penalty_model` function.

.. _scipy.optimize: https://docs.scipy.org/doc/scipy/reference/optimize.html

.. index-end-marker

Installation
------------

.. installation-start-marker

To install:

.. code-block:: bash

    pip install penaltymodel-lp

To build from souce:

.. code-block:: bash
    
    cd penaltymodel_lp
    pip install -r requirements.txt
    python setup.py install

.. installation-end-marker


License
-------

Released under the Apache License 2.0
