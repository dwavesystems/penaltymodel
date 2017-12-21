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

install_requires = ['six',
                    'dwave_networkx>=0.6.0',
                    'pysmt',
                    'penaltymodel==1.0.0.dev3']
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
