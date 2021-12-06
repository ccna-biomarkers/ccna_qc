ccna_qc
=======

QC summary metrics and automatic inspection for [CCNA](https://ccna-ccnv.ca/)

Abstract
--------
Automatic quality inspection on anatomical and functional data has been performed using our internal BIDS app (https://github.com/ccna-biomarkers/ccna_qc).

For each of the 385 subjects, the mean framewise displacement (FDS) was extracted before and after scrubbing (with a threshold of 0.5 mm). As we can see with figure 1 (https://github.com/ccna-biomarkers/ccna_qc/blob/main/reports/figures/raw_fds_mean.png) and figure 2 (https://github.com/ccna-biomarkers/ccna_qc/blob/main/reports/figures/scrubbed_fds_mean.png), scrubbing effectively reduce the mean FDS as expected. We also computed Sørensen–Dice coefficient (REF) between the functional mask and a group fMRI mask evaluated with NiLearn (REF) on all functional runs. For anatomical runs, the dice coefficient was determined between the original MNI template (REF MNI template) and the anatomical mask.

Then, a subject was marked as "fail" if the mean FDS was more than 0.3 mm, anatomical dice was less than 0.99 or functional dice less than 0.89. A summary figure (https://github.com/ccna-biomarkers/ccna_qc/blob/main/reports/figures/dice.png) shows dice for each anatomical and functional run.


Project Organization
--------------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Where the dataset will be installed
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
