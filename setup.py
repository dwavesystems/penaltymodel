from setuptools import setup

from penaltymodel_cache.package_info import __version__, __author__, __description__, __authoremail__

install_requires = ['penaltymodel>=0.9.1',
                    'six',
                    'homebase>=1.0.0']
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
