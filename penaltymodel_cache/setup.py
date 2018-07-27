from __future__ import absolute_import

import sys
from setuptools import setup

_PY2 = sys.version_info.major == 2

# add __version__, __author__, __authoremail__, __description__ to this namespace
if _PY2:
    execfile("./penaltymodel/cache/package_info.py")
else:
    exec(open("./penaltymodel/cache/package_info.py").read())

install_requires = ['penaltymodel>=0.15.0,<0.16.0',
                    'six>=1.11.0,<2.0.0',
                    'homebase>=1.0.0,<2.0.0',
                    'dimod>=0.6.0,<0.7.0'
                    ]

extras_require = {}

packages = ['penaltymodel',
            'penaltymodel.cache',
            ]

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    ]

python_requires = '>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*'

setup(
    name='penaltymodel-cache',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    long_description=open('README.rst').read(),
    url='https://github.com/dwavesystems/penaltymodel',
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=classifiers,
    python_requires=python_requires,
    entry_points={'penaltymodel_factory': ['maxgap = penaltymodel.cache:get_penalty_model'],
                  'penaltymodel_cache': ['penaltymodel_cache = penaltymodel.cache:cache_penalty_model']},
    zip_safe=False
)
