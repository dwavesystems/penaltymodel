from __future__ import absolute_import

import sys
from setuptools import setup

_PY2 = sys.version_info.major == 2

# add __version__, __author__, __authoremail__, __description__ to this namespace
if _PY2:
    execfile("./penaltymodel/mip/package_info.py")
else:
    exec(open("./penaltymodel/mip/package_info.py").read())

install_requires = ['dimod>=0.6.0,<0.7.0',
                    'networkx>=2.0,<3.0',
                    'ortools>=6.6.4659,<7.0.0',
                    'penaltymodel>=0.15.0,<0.16.0',
                    ]

packages = ['penaltymodel',
            'penaltymodel.mip',
            ]

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    ]

python_requires = '>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*'

setup(
    name='penaltymodel-mip',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    long_description=open('README.rst').read(),
    url='https://github.com/dwavesystems/penaltymodel',
    license='Apache 2.0',
    packages=packages,
    classifiers=classifiers,
    python_requires=python_requires,
    install_requires=install_requires,
    entry_points={'penaltymodel_factory': ['mip = penaltymodel.mip:get_penalty_model']},
    zip_safe=False
)
