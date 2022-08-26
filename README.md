mrna
==============================

mcmc version of the mrna model in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5081097/

# How to run the analysis

To run the analysis, run the command `make analysis` from the project root. This
will install a fresh virtual environment if one doesn't exist already, activate
it and install python dependencies and cmdstan, then run the analysis with the
following commands:

- `python prepare_data.py`
- `python sample.py`
- `jupyter execute investigate.ipynb`
