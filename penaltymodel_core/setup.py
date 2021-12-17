import sys
from setuptools import setup

# add __version__, __author__, __authoremail__, __description__ to this namespace
exec(open("./penaltymodel/core/package_info.py").read())

install_requires = ['dimod>=0.10.0,<0.11.0',
                    'networkx>=2.4,<3.0',
                    'numpy>=1.19.1',
                    'scipy>=1.5.2',
                    ]

extras_require = {'all': []}

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
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
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
