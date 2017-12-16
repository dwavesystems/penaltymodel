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

todo - pysmt install instructions


License
-------

Released under the Apache License 2.0. See LICENSE

.. _penaltymodel: https://github.com/dwavesystems/penaltymodel