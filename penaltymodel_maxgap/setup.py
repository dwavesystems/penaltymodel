import sys
import shutil
import tempfile

from setuptools import setup
from setuptools.command.install import install

_PY2 = sys.version_info.major == 2

# add __version__, __author__, __authoremail__, __description__ to this namespace
# equivalent to:
# from penaltymodel_cache.packaing_info import *
if _PY2:
    execfile("./penaltymodel_maxgap/package_info.py")
else:
    exec(open("./penaltymodel_maxgap/package_info.py").read())


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


setup_requires = ['pysmt>=0.7.0,<0.8.0']
install_requires = ['six>=1.11.0,<2.0.0',
                    'dwave_networkx>=0.6.0,<0.7.0',
                    'pysmt>=0.7.0,<0.8.0',
                    'penaltymodel>=0.14.0,<0.15.0',
                    'dimod>=0.6.3,<0.7.0']
extras_require = {}

packages = ['penaltymodel_maxgap']

setup(
    name='penaltymodel_maxgap',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    license='Apache 2.0',
    packages=packages,
    setup_requires=setup_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={'penaltymodel_factory': ['maxgap = penaltymodel_maxgap:get_penalty_model']},
    cmdclass={'install': PysmtSolverInstall},
    zip_safe=False
)
