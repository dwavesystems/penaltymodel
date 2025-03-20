.. _penaltymodel_api_ref:

=============
API Reference
=============

.. currentmodule:: penaltymodel

This package implements the generation and caching of :term:`penalty model`\ s.
The main function for penalty models is :func:`get_penalty_model`. In addition,
the package provides some more-advanced interfaces.

Function
--------

.. autofunction:: get_penalty_model

Cache
-----

.. autoclass:: PenaltyModelCache

Methods
~~~~~~~

.. autosummary::
    :toctree: generated/

    PenaltyModelCache.close
    PenaltyModelCache.insert_binary_quadratic_model
    PenaltyModelCache.insert_graph
    PenaltyModelCache.insert_penalty_model
    PenaltyModelCache.insert_sampleset
    PenaltyModelCache.iter_binary_quadratic_models
    PenaltyModelCache.iter_graphs
    PenaltyModelCache.iter_penalty_models
    PenaltyModelCache.iter_samplesets
    PenaltyModelCache.retrieve

Exceptions
----------

.. autosummary::
    :toctree: generated/

    ImpossiblePenaltyModel
    MissingPenaltyModel

Utilities
---------

.. autosummary::
    :toctree: generated/

    as_graph