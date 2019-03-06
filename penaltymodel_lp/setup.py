from __future__ import absolute_import

import sys
from setuptools import setup

_PY2 = sys.version_info.major == 2

if _PY2:
    execfile("./penaltymodel/lp/package_info.py")
else:
    exec(open("./penaltymodel/lp/package_info.py").read())

FACTORY_ENTRYPOINT = 'penaltymodel_factory'

install_requires = ['dimod>=0.6.0,<0.9.0',
                    'penaltymodel>=0.16.0,<0.17.0',
                    'scipy>=0.15.0,<2.0.0',
                    'numpy>=0.0.0,<1.16.0'
                    ]

packages = ['penaltymodel',
            'penaltymodel.lp',
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
    name="penaltymodel-lp",
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
    entry_points={FACTORY_ENTRYPOINT: ['lp = penaltymodel.lp:get_penalty_model']},
    zip_safe=False
)
