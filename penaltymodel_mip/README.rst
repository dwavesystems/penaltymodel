.. image:: https://img.shields.io/pypi/v/penaltymodel-mip.svg
    :target: https://pypi.python.org/pypi/penaltymodel-mip

Penalty Model - Mixed-Integer (Linear) Programming
==================================================

.. index-start-marker

Generates penalty models using `Google Optimization Tools`_' Mixed-Integer Programming capability.
Serves as a factory and cache for penaltymodel.

On install, penaltymodel-mip registers an entry point that can be read by
penaltymodel. It will be used automatically by any project that uses penaltymodel's
:code:`get_penalty_model` function.

.. _Google Optimization Tools : https://developers.google.com/optimization/

.. index-end-marker

Installation
------------

.. installation-start-marker

To install:

.. code-block:: bash

    pip install penaltymodel-mip

To build from souce:

.. code-block:: bash
    
    cd penaltymodel_mip
    pip install -r requirements.txt
    python setup.py install

.. installation-end-marker


License
-------

Released under the Apache License 2.0
