from pkg_resources import iter_entry_points


FACTORY = 'penaltymodel_factory'
CACHE = 'penaltymodel_cache'


def entry_point(priority):
    def _entry_point(f):
        f.priority = priority
        return f
    return _entry_point


def factories():
    """TODO"""
    # retrieve all of the factories with
    factories = (entry.load() for entry in iter_entry_points(FACTORY))

    # sort the factories from highest priority to lowest. Any factory with unknown priority
    # gets assigned priority -1000.
    return sorted(factories, key=lambda f: getattr(f, 'priority', -1000), reverse=True)


def caches():
    """TODO"""
    # for caches we don't need an order
    return [entry.load() for entry in iter_entry_points(CACHE)]
