from loguru import logger


def display_available_files(available_files, file_pattern):
    """Display the available pipelines to the user."""
    logger.debug(f"Available {file_pattern} Files:")
    for idx, file in enumerate(available_files, 1):
        logger.debug(f"{idx}. {file}")


def get_user_choice(available_files, pattern):
    """Get a valid user choice with retry mechanism."""
    while True:
        try:
            choice = int(input(f"\nEnter the number of the {pattern} files you want to run: "))
            if choice < 1 or choice > len(available_files):
                logger.warning("Invalid choice. Please select a valid option.")
            else:
                return available_files[choice - 1]
        except ValueError:
            logger.warning("Invalid input. Please enter a valid number.")


def confirm_file_selection(file_name: str) -> bool:
    """Confirm the selected pipeline_old."""
    confirmation = input(f"You selected {file_name}. Do you want to proceed? (y/n): ")
    return confirmation.lower() == 'y'
