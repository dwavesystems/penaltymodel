
class FactoryException(Exception):
    """General exception for a factory being not able to produce a penalty model."""


class ImpossibleSpecification(FactoryException):
    """"""


class MissingPenaltyModel(FactoryException):
    """"""
