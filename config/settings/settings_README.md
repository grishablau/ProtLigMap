# settings/

**Note:** This folder is currently under construction and not fully integrated into the project yet. It is being set up as the central place for configuration and will be used more extensively as the project evolves.


This folder contains configuration files for the project, organized by purpose and component. It includes both:

- **YAML files** — for declarative configuration of pipelines, models, paths, etc.
- **Python modules** — for dynamic or programmatic settings when logic is required.

## Purpose

The `settings/` directory centralizes all project-specific settings to improve maintainability, consistency, and visibility of key parameters.

Configurations here are used to:

- Control preprocessing and model pipeline behavior
- Store paths, flags, and runtime parameters
- Separate environment- or task-specific settings (via subfolders)

## Future Use

This folder is intended to **scale with the project**. As the project evolves, new configurations, environment profiles (e.g., dev/prod), or plugin/module-specific settings can be added here.

Keeping this folder organized and well-documented will make future expansion easier and safer.



