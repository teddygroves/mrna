mrna
==============================

mcmc version of the mrna model in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5081097/

# How to install dependencies

Run this command from the project root:

```
pip install -r requirements.txt
install_cmdstan
```

# How to run the analysis

To run the analysis, run the command `make analysis` from the project root.

This will run the following commands

- `python prepare_data.py`
- `python sample.py`
- `jupyter execute investigate.ipynb`



# How to run tests

Run this command from the project root:

```
python -m pytest
```

