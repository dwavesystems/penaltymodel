import os
import sys

import penaltymodel_cache as pmc

version = pmc.__version__

tag = os.getenv('CIRCLE_TAG')

if tag != 'penaltymodel-cache {}'.format(version):
    sys.exit("Git tag: {}, expected: penaltymodel-cache {}".format(tag, version))
