import sys
from setuptools import setup

_PY2 = sys.version_info.major == 2

# add __version__, __author__, __authoremail__, __description__ to this namespace
if _PY2:
    execfile("./penaltymodel/core/package_info.py")
else:
    exec(open("./penaltymodel/core/package_info.py").read())

install_requires = ['dimod>=0.6.3,<0.7.0',
                    'six>=1.11.0,<2.0.0',
                    'networkx>=2.0,<3.0',
                    'enum34>=1.1.6,<2.0.0'
                    ]

extras_require = {'all': ['penaltymodel_cache>=0.3.0,<0.4.0',
                          'penaltymodel_maxgap>=0.4.0,<0.5.0',
                          "penaltymodel_mip>=0.1.0,<0.2.0; platform_machine!='x86' and python_version!='3.4'"
                          ]
                  }

packages = ['penaltymodel',
            'penaltymodel.core',
            'penaltymodel.core.classes'
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
    name='penaltymodel',
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
    extras_require=extras_require,
    zip_safe=False
)
