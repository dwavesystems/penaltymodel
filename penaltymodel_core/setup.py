import sys
from setuptools import setup

_PY2 = sys.version_info.major == 2

# add __version__, __author__, __authoremail__, __description__ to this namespace
if _PY2:
    execfile("./penaltymodel/package_info.py")
else:
    exec(open("./penaltymodel/package_info.py").read())

install_requires = ['dimod>=0.6.3,<0.7.0',
                    'six>=1.11.0,<2.0.0',
                    'networkx>=2.0,<3.0',
                    'enum34>=1.1.6,<2.0.0']
extras_require = {'all': ['penaltymodel_cache>=0.2.1,<0.3.0',
                          'penaltymodel_maxgap>=0.3.0,<0.4.0']}

packages = ['penaltymodel',
            'penaltymodel.classes']

setup(
    name='penaltymodel',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    url='https://github.com/dwavesystems/penaltymodel',
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require
)
