from __future__ import absolute_import

import sys
from setuptools import setup

_PY2 = sys.version_info.major == 2

# add __version__, __author__, __authoremail__, __description__ to this namespace
# equivalent to:
# from penaltymodel_cache.packaing_info import *
if _PY2:
    execfile("./penaltymodel_cache/package_info.py")
else:
    exec(open("./penaltymodel_cache/package_info.py").read())

install_requires = ['penaltymodel>=0.14.0,<0.15.0',
                    'six>=1.11.0,<2.0.0',
                    'homebase>=1.0.0,<2.0.0',
                    'dimod>=0.6.0,<0.7.0']

extras_require = {}

packages = ['penaltymodel_cache']

setup(
    name='penaltymodel_cache',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    url='https://github.com/dwavesystems/penaltymodel_cache',
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={'penaltymodel_factory': ['maxgap = penaltymodel_cache:get_penalty_model'],
                  'penaltymodel_cache': ['penaltymodel_cache = penaltymodel_cache:cache_penalty_model']}
)
