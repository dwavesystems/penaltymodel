import sys
import shutil
import tempfile

from setuptools import setup
from setuptools.command.install import install

# add __version__, __author__, __authoremail__, __description__ to this namespace
exec(open("./penaltymodel/maxgap/package_info.py").read())


class PysmtSolverInstall(install):
    """Custom install command that after standard install, runs install
    of all pySMT solvers required, in the current environment.
    """

    def run(self):
        install.run(self)

        # available only after setup requirements installed in previous step
        from pysmt.cmd.install import INSTALLERS

        # install z3 into current environment's site-packages dir
        required_solvers = ['z3']
        bindings_dir = self.install_lib
        force_redo = False

        # use temp dir for z3 build by pysmt installer
        install_dir = tempfile.mkdtemp()

        for i in INSTALLERS:
            name = i.InstallerClass.SOLVER
            if name in required_solvers:
                installer = i.InstallerClass(
                    install_dir=install_dir, bindings_dir=bindings_dir,
                    solver_version=i.version, **i.extra_params)
                installer.install(force_redo=force_redo)

        shutil.rmtree(install_dir)


setup_requires = ['pysmt==0.8.0']

install_requires = ['dimod>=0.8.0,<0.10.0',
                    'dwave_networkx>=0.6.0',
                    'penaltymodel>=0.16.0,<0.17.0',
                    'pysmt==0.8.0',
                    ]

extras_require = {}

packages = ['penaltymodel',
            'penaltymodel.maxgap',
            ]

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9'
    ]

python_requires = '>=3.5'

setup(
    name='penaltymodel-maxgap',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    long_description=open('README.rst').read(),
    license='Apache 2.0',
    packages=packages,
    setup_requires=setup_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=classifiers,
    python_requires=python_requires,
    entry_points={'penaltymodel_factory': ['maxgap = penaltymodel.maxgap:get_penalty_model']},
    cmdclass={'install': PysmtSolverInstall},
    zip_safe=False
)
