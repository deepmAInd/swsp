"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from swsp.pipelines.preprocess import create_pipeline as create_preprocess


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {"__default__": create_preprocess()}
