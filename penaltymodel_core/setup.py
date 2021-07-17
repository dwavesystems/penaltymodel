import sys
from setuptools import setup

# add __version__, __author__, __authoremail__, __description__ to this namespace
exec(open("./penaltymodel/core/package_info.py").read())

install_requires = ['dimod>=0.8.0,<0.11.0',
                    'six>=1.11.0,<2.0.0',
                    'networkx>=2.4,<3.0'
                    ]

extras_require = {'all': ['penaltymodel_cache>=0.3.0,<0.4.0',
                          'penaltymodel_lp>=0.1.0,<0.2.0',
                          'penaltymodel_maxgap>=0.5.0,<0.6.0',
                          "penaltymodel_mip>=0.2.0,<0.3.0; platform_machine!='x86' and python_version!='3.4'"
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
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9'
    ]

python_requires = '>=3.6'

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
