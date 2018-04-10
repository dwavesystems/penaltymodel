import sys
from setuptools import setup

_PY2 = sys.version_info.major == 2

# add __version__, __author__, __authoremail__, __description__ to this namespace
# equivalent to:
# from penaltymodel_cache.packaing_info import *
if _PY2:
    execfile("./penaltymodel_maxgap/package_info.py")
else:
    exec(open("./penaltymodel_maxgap/package_info.py").read())

install_requires = ['six>=1.11.0,<2.0.0',
                    'dwave_networkx>=0.6.0,<0.7.0',
                    'pysmt>=0.7.0,<0.8.0',
                    'penaltymodel>=0.13.0,<0.14.0',
                    'dimod>=0.6.3,<0.7.0']
extras_require = {}

packages = ['penaltymodel_maxgap']

setup(
    name='penaltymodel_maxgap',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={'penaltymodel_factory': ['maxgap = penaltymodel_maxgap:get_penalty_model']}
)
