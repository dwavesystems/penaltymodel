from setuptools import setup

from penaltymodel.plugins import FACTORY, CACHE

from penaltymodel_cache import __version__

install_requires = ['penaltymodel',
                    'networkx>=2.0']
extras_require = {}

packages = ['penaltymodel_cache']

setup(
    name='penaltymodel_cache',
    version=__version__,
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={FACTORY: ['maxgap = penaltymodel_cache:get_penalty_model'],
                  CACHE: ['penaltymodel_cache = penaltymodel_cache:cache_penalty_model']}
)
