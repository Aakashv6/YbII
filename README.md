# YbII Experiment Overview

## Introduction
YbII builds on top of a "standard" Rydberg atom array experiment using **171Yb** with an optical cavity to access the strong atom-photon coupling regime of cavity QED. We will integrate a Rydberg atom array optical clock with the optical cavity to access a host of new opportunities. This system will also contain in-vacuum electrodes that mostly serve the purpose of shielding the atoms against electric fields from the PZT stacks, but may also serve more exotic purposes.

## Top-Level Directory Structure
- **`/docs`** - Documentation for the project, including setup instructions, usage guides, and methodological notes.
- **`/data`** - Data files, typically kept empty in the repo with a `.gitignore` file to exclude actual data files from being uploaded, but include sample data or scripts to download data.
- **`/src`** - Source code for your data collection and analysis scripts.
- **`/tests`** - Test cases and testing scripts to ensure your code works as expected.
- **`/notebooks`** - Jupyter notebooks or other interactive tools for demonstration, training, or exploratory data analysis.
- **`/results`** - For storing output from analyses, like tables or figures, which can be version-controlled if they are summaries or key results.
- **`/bin`** - For any compiled binaries or scripts that are meant to be run directly.
- **`/config`** - Configuration files or environment specifics.

## Key Files
- **`README.md`** - Overview of the project, setup instructions, contributor list, and usage examples.
- **`.gitignore`** - Specifies intentionally untracked files that Git should ignore (e.g., sensitive credentials, large data files).
- **`LICENSE`** - The license under which the project is released.
- **`requirements.txt`** or **`environment.yml`** - List all dependencies for Python projects, which could be used with pip or conda, respectively, to set up virtual environments.
- **`setup.py`** (if applicable) - If your codebase is to be installed as a package, this file will handle the installation.

## Ignored Data Formats
For the purpose of maintaining a clean and manageable repository, certain data formats are specifically excluded from tracking in our Git repository. Currently, the `.gitignore` file is configured to ignore the following data formats:
- **`*.png`**
- **`*.jpg`**
- **`*.jpeg`**
- **`*.bmp`**
- **`*.pyc`**
- **`*.ipynb_checkpoints`**

This setup helps to ensure that our repository remains efficient to clone and pull, without overloading it with large or transient data files that do not contribute to the core functionality of the project.