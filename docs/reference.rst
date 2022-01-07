.. _reference_penaltymodel:

Reference Documentation
***********************

.. currentmodule:: penaltymodel

This package implements the generation and caching of :term:`penalty model`\ s.

The main function for penalty models is:

.. autofunction:: get_penalty_model

In addition to :func:`get_penalty_model`, there are some more advanced
interfaces available.

Exceptions
==========

.. autosummary::
    :toctree: generated/

    ImpossiblePenaltyModel
    MissingPenaltyModel

Utilities
=========

.. autosummary::
    :toctree: generated/

    as_graph
