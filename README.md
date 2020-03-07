# Manifolder

v0.0.2

Python implementation of "Empirical Intrinsic Geometry" (EIG) code, for machine learning multivariate time-series.

The algorithm was originally described in the 2014 paper "Intrinsic modeling of stochastic dynamical systems using empirical geometry" by Talmon and Coifman) ([link to pdf](https://ronentalmon.com/wp-content/uploads/2019/03/ACHA_EIG.pdf)), and ported to Python from the original MATLAB code, written by Ronen Talmon (2013).

### Installing and Running

To see a quick demo of the code, check out the [manifolder_notebook](https://github.com/avlab/manifolder/blob/master/manifolder_notebook.ipynb), which can be view inline on the web, and gives an idea of what the program can do.

The code can also be downloaded and run locally.  Note, we are in the process of reformatting the code, to make it much easier to work with; ideally running `manifolder` have the some syntax as running [Python's built-in clustering models](https://scikit-learn.org/stable/modules/clustering.html).  For now, you can view the notebook directly in github, or run locally with these steps (starting from scratch):

* Install a recent Python distribution, currently Python 3.7.  We recommend the [Anaconda distribution](https://www.anaconda.com/distribution/#download-section), which contains most of the packages useful for data science.
* Download or clone [the manifolder](https://github.com/avlab/manifolder) repository.  Note the repository is private, and you must be logged into github, and granted permission], too see it.
* Make sure additional needed libraries are installed.  Open a terminal, go to the root of the repository, and install additional components with `pip install -r requirements.txt` ... likely some packages will still need to be added to the list.
* Add the location of the Python files to the `PYTHONPATH` environment variable (instructions are in the manifolder notebookS)

When you are setup, start a jupyter notebook server, and run the `manifolder_notebook.pynb`. 



