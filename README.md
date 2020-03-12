# Manifolder

v0.0.2

Python implementation of "Empirical Intrinsic Geometry" (EIG) code, for machine learning multivariate time-series.

The algorithm was originally described in the 2014 paper "Intrinsic modeling of stochastic dynamical systems using empirical geometry" by Talmon and Coifman) ([link to pdf](https://ronentalmon.com/wp-content/uploads/2019/03/ACHA_EIG.pdf)), and ported to Python from the original MATLAB code, written by Ronen Talmon (2013).

This port is open-sourced under the MIT license.

### Installing and Running

To see a quick demo of the code, check out the [manifolder_notebook](https://github.com/avlab/manifolder/blob/master/manifolder_notebook.ipynb), which can be view inline on the web, and gives an idea of what the program can do.

The code can also be downloaded and run locally.  Note, we are in the process of reformatting the code, to make it much easier to work with; ideally running `manifolder` have the some syntax as running [Python's built-in clustering models](https://scikit-learn.org/stable/modules/clustering.html).  For now, you can view the notebook directly in github, or run locally with these steps (starting from scratch):

* Install a recent Python distribution, currently Python 3.7.  We recommend the [Anaconda distribution](https://www.anaconda.com/distribution/#download-section), which contains most of the packages useful for data science.
* Download or clone [the manifolder](https://github.com/avlab/manifolder) repository.  Note the repository is private, and you must be logged into github, and granted permission], too see it.
* Make sure additional needed libraries are installed.  Open a terminal, go to the root of the repository, and install additional components with `pip install -r requirements.txt` ... likely some packages will still need to be added to the list.
* Add the location of the Python files to the `PYTHONPATH` environment variable (instructions are in the manifolder notebookS)

When you are setup, start a jupyter notebook server, and run the `manifolder_notebook.pynb`. 

### Porting Notes

Package implements the algorithm described in the paper *2014 Talmon and Coifman - Intrinsic modeling of stochastic dynamical systems using empirical geometry* [.pdf](https://ronentalmon.com/wp-content/uploads/2019/03/ACHA_EIG.pdf)

Initial code is essentially a port of the original MATLAB code.  To make the ported code as clear as possible, the python translation strives to keep most of the original variable names and structure.  For instance, time series in the MATLAB code is encoded as rows in `z`, and the orientation is the same in the Python code.

Indexing starts at zero in Python, and one in MATLAB.  The port adopts the correct Python indexing (starting at one), to prevent confusion for Python users of the library.  For the most part, this change is almost transparent, since both languages keep their native idioms.  For instance, a MATLAB loop like

```octave
# MATLAB Loop

for dim=1:N
	# do something
end
```

is replaced by the equivalent 

```python
# Python Loop

for dim in range(N):  # note, loop will run in standard Python, starting at dim = 0
    # do something
```

### Code and Package Structure

Since the idea is to distribute the package, we are using the standard python package structure discussed at [Packaging Python Projects — Python Packaging User Guide](https://packaging.python.org/tutorials/packaging-projects/).  Python package management is one of the major advantages of the language.  The software piece `pip` allows most python packages to be installed in one line, while keeping track of package dependencies, and updating them as necessary.

```bash
pip install --index-url https://hosturl.org/simple/ example-pkg-YOUR-USERNAME-HERE
```

Ideally, the Python code should run using the coding interface defined by the industry standard [scikit-learn](https://scikit-learn.org/stable/) Python package.  However, on the initial port, the code needs to retain most of the structure of the original MATLAB code, to make sure the port was done correctly.

A later phase of the project can refactor the code, and compare the results using existing datasets.
