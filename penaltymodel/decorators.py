__all__ = ['entry_point']


def entry_point(priority):
    def _entry_point(f):
        f.priority = priority
        return f
    return _entry_point
