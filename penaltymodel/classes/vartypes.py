import enum

__all__ = ['VARTYPES', 'SPIN', 'BINARY', 'UNDEFINED']


class VARTYPES(enum.Enum):
    """An enumeration of the types of variables for the binary quadratic model.

    Examples:
        >>> vartype = VARTYPES.SPIN
        >>> print(vartype)
        'VARTYPES.SPIN'
        >>> isinstance(vartype, VARTYPES)
        True
        >>> vartype.value
        frozenset({-1, 1})

        Access can also be by value or name
        >>> print(VARTYPES({0, 1}))
        'VARTYPES.BINARY'
        >>> print(VARTYPES['SPIN'])
        'VARTYPES.SPIN'

    """
    SPIN = frozenset({-1, 1})
    BINARY = frozenset({0, 1})
    UNDEFINED = None

SPIN = VARTYPES.SPIN
BINARY = VARTYPES.BINARY
UNDEFINED = VARTYPES.UNDEFINED
