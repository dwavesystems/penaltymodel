import os
import sys

import penaltymodel_cache as pmc

version = pmc.__version__

tag = os.getenv('CIRCLE_TAG')

if tag != 'penaltymodel-maxgap {}'.format(version):
    sys.exit("Git tag: {}, expected: penaltymodel-maxgap {}".format(tag, version))
