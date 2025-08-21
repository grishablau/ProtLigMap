from config.settings.data_loader.data_config import config

# Call the function to load data and assign to 'result' to pass back to pipeline runner
result = data_preproc.data_loader.load_project_data(config)
