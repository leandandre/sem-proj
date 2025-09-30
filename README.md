# Semester Project - Machine Learning applied to brain's EEG signals
This repository contains the code and data for my Semester Project.


## Project Structure

sem_proj/
    configs/ # Experiments congfig (YAML)
    data/ # Datasets (Ignored by Git)
        raw/ # Original Data
        intermediate/ # Intermediate data
        processed/ # Final training data
    models/ # Saved checkpoints (ignored in Git)
    notebooks/ # Jupyter notebooks
    reports/ # Results and outputs
        figures/ # Plots/images (ignored in Git)
        metrics/ # Logs/metrics (ignored in Git)
    src/sem-proj/ # Source code
        data/ # Data loading
        features/ # Feature engineering
        models/ # Model definitions
        training/ # Training loop
        evaluation/ # Evaluation scripts
        inference/ # Inference resp prediction
        utils/ # Helpers
    scripts/ # CLI entry points
    tests/ # Unit tests
