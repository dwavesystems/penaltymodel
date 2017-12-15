from setuptools import setup

from penaltymodel_maxgap.packge_info import __version__

install_requires = []
extras_require = {}

packages = ['penaltymodel_maxgap']

setup(
    name='penaltymodel_maxgap',
    version=__version__,
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={'penaltymodel_factory': ['maxgap = penaltymodel_maxgap:get_penalty_model']}
)
