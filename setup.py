from setuptools import setup

from penaltymodel_maxgap.package_info import __version__, __author__, __description__, __authoremail__

install_requires = ['six', 'dwave_networkx>=0.6.0', 'pysmt', 'penaltymodel>=0.9.1']
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
