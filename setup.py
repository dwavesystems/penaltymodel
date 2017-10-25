from setuptools import setup

from penaltymodel import __version__

install_requires = []
extras_require = {}

packages = ['penaltymodel']

setup(
    name='penaltymodel',
    version=__version__,
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require
)
