.. image:: https://travis-ci.org/dwavesystems/penaltymodel_cache.svg?branch=master
    :target: https://travis-ci.org/dwavesystems/penaltymodel_cache

.. image:: https://readthedocs.org/projects/penaltymodel-cache/badge/?version=latest
    :target: http://penaltymodel-cache.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/dwavesystems/penaltymodel_cache/badge.svg?branch=master
    :target: https://coveralls.io/github/dwavesystems/penaltymodel_cache?branch=master

.. inclusion-marker-do-not-remove

Penalty Model Cache
===================

A local cache for penalty models. Serves as a factory and cache for penaltymodel_.

How it works
------------

On install, penaltymodel_cache registers an entry point that can be read by
penaltymodel. By identifying itself as both a cache and a factory, it will
be used automatically by any project that uses penaltymodel's :code:`get_penalty_model`
function. It will also be automatically populated

Installing
----------

To install:

.. code-block:: bash

    pip install penaltymodel_cache

To build from source:

.. code-block:: bash

    git clone https://github.com/dwavesystems/penaltymodel_cache.git
    cd penaltymodel_cache
    python setup.py install

Cache Location
--------------

If installed inside of a virtual environment, penaltymodel_cache will
store the database in :code:`/path/to/virtualenv/data/app_name`. Otherwise
the cache will be placed in the system's application data directory.

License
-------

Released under the Apache License 2.0. See LICENSE

.. _penaltymodel: https://github.com/dwavesystems/penaltymodel
