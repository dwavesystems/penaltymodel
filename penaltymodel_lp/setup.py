from setuptools import setup

FACTORY_ENTRYPOINT = 'penaltymodel_factory'

setup(
    name="penaltymodel-lp",
    entry_points={
        FACTORY_ENTRYPOINT: ['lp = penaltymodel.lp:get_penalty_model']
    }
)