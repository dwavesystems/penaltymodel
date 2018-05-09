.. image:: https://img.shields.io/pypi/v/penaltymodel-cache.svg
    :target: https://pypi.python.org/pypi/penaltymodel-cache

Penalty Model Cache
===================

.. index-start-marker

A local cache for penalty models. Serves as a factory and cache for penaltymodel.

On install, penaltymodel_cache registers an entry point that can be read by
penaltymodel. By identifying itself as both a cache and a factory, it will
be used automatically by any project that uses penaltymodel's :code:`get_penalty_model`
function. It will also be automatically populated

.. index-end-marker

Installing
----------

.. installation-start-marker

To install:

.. code-block:: bash

    pip install penaltymodel_cache

To build from source:

.. code-block:: bash

    cd penaltymodel_cache
    pip install -r requirements.txt
    pip install -e ../penaltymodel_core/
    python setup.py install

.. installation-end-marker

Cache Location
--------------

If installed inside of a virtual environment, penaltymodel_cache will
store the database in :code:`/path/to/virtualenv/data/app_name`. Otherwise
the cache will be placed in the system's application data directory.

License
-------

Released under the Apache License 2.0. See LICENSE
