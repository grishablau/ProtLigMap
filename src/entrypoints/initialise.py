from src.core_objects.env_obj import Environment
from src.core_objects.instsystem_obj import InstSystem
import os
import sys
from loguru import logger


def initialise_env():
    """The main and first function to initialise the basic env settings"""
    return Environment()


def initialise_pipeline(env):
    """Initialise pipeline from env"""
    # Initialise pipeline object
    return InstSystem(env)


def initialize_pipeline_for_execution(env):
    """Initialise pipeline from env and then prepare it all the way to be ready to run"""
    # Initialise basics of the pipeline
    pipeline = initialise_pipeline(env)
    # Prepare it for run
    pipeline.pre_run_proc()
    return pipeline


def initialise():
    # Step 1 - Initialise the env variables
    env = initialise_env()
    # Check if ENV_MODE is set to PIPELINE. if not raise an error since we still don't have other instruction settings.
    # If True, continue to setting the pipeline
    if os.getenv('ENV_MODE', '').upper() == "PIPELINE":
        return initialize_pipeline_for_execution(env)
    else:
        logger.error("Error: 'pipeline' is the only ENV_MODE for now. Exiting program.")
        sys.exit(1)  # Exit the program with a non-zero exit status indicating an error
