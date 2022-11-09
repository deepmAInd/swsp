"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import make_patient_data, combine_patients


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=make_patient_data,
                inputs=["patient_raw", "patient_int@path"],
                outputs=None,
                name="create-patient-data",
            ),
            node(
                func=combine_patients,
                inputs=["patient_int@PartitionedDataSet"],
                outputs="patient_feature",
                name="combine-patient-data",
            ),
        ]
    )
