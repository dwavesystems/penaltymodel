# extras_require = {'tests': ['networkx>=2.0']}
from setuptools import setup

from dwave_maxgap import __version__

install_requires = []
extras_require = {}

packages = ['dwave_maxgap']

setup(
    name='dwave_maxgap',
    version=__version__,
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={'penalty_model_factory': ['maxgap = dwave_maxgap:get_penalty_model']}
)
