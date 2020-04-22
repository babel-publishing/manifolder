# Manifolder

v0.0.3

Python implementation of "Empirical Intrinsic Geometry" (EIG) code, for machine learning multivariate time-series.

The algorithm was originally described in the 2014 paper "Intrinsic modeling of stochastic dynamical systems using empirical geometry" by Talmon and Coifman) ([link to pdf](https://ronentalmon.com/wp-content/uploads/2019/03/ACHA_EIG.pdf)), and ported to Python from the [original MATLAB](http://www.runmycode.org/companion/view/191) code, written by Ronen Talmon (2013).

This port is open-sourced under the MIT license.

### Overview

Manifolder uses the sklearn-like interface.  In the simplest case, data is loaded as a time series, where the rows are the time steps, and the columns are the features.  (This is the standard Python data format; note that inside the Manifolder class, the data is stored transposed, with columns as time, for compatibility with the original code).

```python
# load the data
data_location = 'data/simple_data.csv'
df = pd.read_csv(data_location, header=None)
z = df.values
print('loaded',data_location + ', shape:', z.shape)

# create manifolder object
manifolder = mr.Manifolder()

# add the data, and fit (this runs all the functions)
# make sure data is loaded as with [time,features] orientation
manifolder.fit_transform( z.T )

manifolder._clustering()  # display
```



The EIG technique used by Manifolder relies on the data in `z` being an unbroken series of observations (i.e., they are a time series).  The manifold can also be constructed using multiple sets of time series.  In this case, the data sent to `fit_transform` should be a list of matrices, like `z = [za, zb, zc, ...]`



### Installing and Running

To see a quick demo of the code, check out the [manifolder_notebook](https://github.com/avlab/manifolder/blob/master/manifolder_notebook.ipynb), which can be view inline on the web, and gives an idea of what the program can do.

The code can also be downloaded and run locally.

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



# Future Work

* Additional Datasets

  * [UCI Machine Learning Repository  Pen-Based Recognition of Handwritten Digits Data Set.html](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits)
  * [UCI Machine Learning Repository  UJI Pen Characters Data Set.html](http://archive.ics.uci.edu/ml/datasets/UJI+Pen+Characters)
* Additional Metrics
  * Can use TSLEAN to add additional distance metrics (**dtw** and **GAK**), [tslearn.metrics](https://tslearn.readthedocs.io/en/latest/gen_modules/tslearn.metrics.html)
  * Also suggested to use Euclidian (is that the default?)



### Background Papers

* [2016 Or Yaira et. al. - No equations, no parameters, no variables  data, and the reconstruction of normal forms by learning informed observation geometries](https://www.researchgate.net/publication/311585902_No_equations_no_parameters_no_variables_data_and_the_reconstruction_of_normal_forms_by_learning_informed_observation_geometries)

* [The UEA multivariate time series classification archive, 2018](https://arxiv.org/pdf/1811.00075.pdf) includes links to multivariate time series datasets.

* [2019 Lin et al - Wave-shape oscillatory model for biomedical time series with applications](https://www.researchgate.net/publication/334161695_Wave-shape_oscillatory_model_for_biomedical_time_series_with_applications) - uses EIG for ECG

* [2020 Liu et al - Diffuse to fuse EEG spectra – Intrinsic geometry of sleep dynamics for classification](https://www.sciencedirect.com/science/article/pii/S1746809419301508)

* [2012 We et al - Assess Sleep Stage by Modern Signal Processing
Techniques](https://arxiv.org/pdf/1410.1013.pdf)

* [2014 Talmon et al - Manifold Learning for Latent Variable Inference in Dynamical Systems](https://cpsc.yale.edu/sites/default/files/files/tr1491.pdf)


### Packaging In GIT

[https://packaging.python.org/tutorials/packaging-projects/](https://packaging.python.org/tutorials/packaging-projects/) is the main resource for creating a package - maybe this can be added to contributing?  Also, include notes on code formatting, etc.?

The code that actually runs is

(update `setuptools` and `wheel`):

```bash
python3 -m pip install --user --upgrade setuptools wheel
```

Now run this command from the same directory where `setup.py` is located:

```bash
python3 setup.py sdist bdist_wheel
```

[maybe better](https://packaging.python.org/guides/distributing-packages-using-setuptools/) `python setup.py bdist_wheel --universal`

This command should output a lot of text and once completed should generate two files in the `dist`directory:

This command should output a lot of text and once completed should generate two files in the `dist`directory:

```
dist/
  example_pkg_YOUR_USERNAME_HERE-0.0.1-py3-none-any.whl
  example_pkg_YOUR_USERNAME_HERE-0.0.1.tar.gz
```

... and also creates `manifolder_pkg_avlab.egg-info/` directory

At this point the distribution is build, and should be uploadable to github and installable.  

### Virtual Environment



https://help.github.com/en/github/creating-cloning-and-archiving-repositories/licensing-a-repository#where-does-the-license-live-on-my-repository



[The Legal Side of Open Source | Open Source Guides](https://opensource.guide/legal/)

[How to Contribute to Open Source | Open Source Guides](https://opensource.guide/how-to-contribute/) - has the "Anatomy of an open source project" section, on the crucial files, as well as the structure (Author, Owner, Maintainers, Contributors, Community Members)

[FOSSmarks](http://fossmarks.org) is a good source for understanding TRADEMARKS in a free and open-source domain


## github.io (web)

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/avlab/avlab.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.





