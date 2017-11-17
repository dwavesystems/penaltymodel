from penaltymodel.plugins import factories, caches
from penaltymodel.exceptions import FactoryException


def get_penalty_model_from_specification(specification):

    # Iterate through the available factories until one gives a penalty model
    for factory in factories():
        try:
            pm = factory(specification)
        except FactoryException:
            continue

        # if penalty model was found, broadcast to all of the caches. This could be done
        # asynchronously
        for cache in caches():
            cache(pm)

        return pm

    return None
