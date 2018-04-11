import os
import sys

import penaltymodel as pm

version = pm.__version__

tag = os.getenv('CIRCLE_TAG')

if tag != 'penaltymodel-core {}'.format(version):
    sys.exit("Git tag: {}, expected: penaltymodel-core {}".format(tag, version))
