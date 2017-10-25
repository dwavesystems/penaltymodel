from pkg_resources import iter_entry_points


def factories():
    """TODO"""
    factories = (entry.load() for entry in iter_entry_points('penalty_model_factory'))

    # sort the factories from highest priority to lowest. Any factory with unknown priority
    # gets assigned priority -1000.
    return sorted(factories, key=lambda f: getattr(f, 'priority', -1000), reverse=True)


def caches():
    raise NotImplementedError
