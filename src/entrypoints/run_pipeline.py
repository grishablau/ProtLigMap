from loguru import logger
from src.utils.file_system import function_loader


def run_pipeline(pipeline):
    function_folder = pipeline.inst_obj.instructions_raw['pipeline']['pipeline functions folder']
    for step in pipeline.inst_obj.steps:
        function = function_loader(step.info, function_folder)
        if callable(function):
            function(step.info['input'])

        else:
            logger.error(f"Error: {function} is not callable.")
