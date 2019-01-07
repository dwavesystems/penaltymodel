from setuptools import setup

FACTORY_ENTRYPOINT = 'penaltymodel_factory'

install_requires = ['dimod>=0.6.0,<0.9.0',
                    'penaltymodel>=0.15.5,<0.16.0',
                    'scipy>=0.15.0,<2.0.0',
                    'numpy>=0.0.0,<1.16.0'
                    ]

packages = ['penaltymodel',
            'penaltymodel.lp',
            ]

setup(
    name="penaltymodel-lp",
    install_requires=install_requires,
    packages=packages,
    entry_points={
        FACTORY_ENTRYPOINT: ['lp = penaltymodel.lp:get_penalty_model']
    }
)