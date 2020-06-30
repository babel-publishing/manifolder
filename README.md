# Manifolder

> (Dear Team: change this if it sucks! Just pushing it without worrying about editing too much.)

## Table of Contents
* [What are we trying to do?](#what-are-we-trying-to-do)
* [What's the paper about?](#whats-the-paper-about)
* [What's the code about?](#whats-the-code-about)

## I want to know more
* [Learn more about the code](code/README.md)
* [Learn more about the paper](paper/README.md)

---

## What are we trying to do?

This paper and the code underlying it are an experiment in scientific collaboration outside the ivory tower.

Our aim is to explore what peer review and publication might look like in a world where science has decoupled itself from academia.

This project will serve as a test drive of a model of scientific publishing with the following core features:

1. Publication happens first. Before peer-review. Before the paper has been written. And before the work backing it has been finished.

2. Peer review begins immediately, and occurs continuously throughout the lifetime of the project. It is not limited to unpublished reviews carried out in secret prior to publication. Peer review in this model is the process of developing the ideas in public, hashing out disagreements in public, asking clarifying questions in public, receiving contributions from curious readers in public, writing the paper together in public, and letting the whole exchange of ideas take place in a manner that is publicly visible and logged, for the reader and for posterity. The software development crowd is miles ahead of the scientific community in the scope and scale of collaboration. It's time we learn from them what peer review means.

This experiment is best served not through a collaboration on some revolutionary new idea, but through a simple collaboration on what we might call a "normal scientific paper."

Aside from these details, our goal is not to push any particular agenda, and we have no unified opinion of how scientific collaboration should be done.

We intend to figure out the details as we go.

If you would like to collaborate on this project, post an issue or submit a pull request.

If you would like to be added to the organization, send an email to notarealdeveloper@gmail.com or any other member.


## What's the paper about?

**Questions we need to answer!**
* Why do we care?
  * Lots of high dimensional real world datasets seem to live on a low dimensional submanifold.
    * For example, images. Sample a random grid of pixels from any distribution. It won't look like a dog.
    * Also audio. A random sound file, however you generate it, won't sound like much.
  * In the Riemannian geometry textbooks, a manifold is something nice and neat delivered to us by the gods.
  * In the real world, we don't get to start with an atlas or coordinate charts or maps from our manifold to $R^n$ or any of that.
  * So it'd be cool if we could somehow *learn* the low-dimensional manifold that a real-world dataset lives on.
* That sounds hard. How far have other folks gotten on this problem?
  * [This paper by Talmon and Coifman did an awesome job](background/intrinsic-modeling-of-stochastic-dynamical-systems-using-empirical-geometry.pdf)
  * We want to implement their algorithm in Python and use it on some space weather data.


## What's the code about?

It's a Python implementation of "Empirical Intrinsic Geometry" (EIG), for machine learning multivariate time-series.

The algorithm was originally described [by Talmon and Coifman] in the 2014 paper "Intrinsic modeling of stochastic dynamical systems using empirical geometry", and ported to Python from the original MATLAB code written by Ronen Talmon (2013).

Our port is open-sourced under the MIT license.

To install the python package, use the following command:
```python
$ pip install -e 'git+https://github.com/babel-publishing/manifolder/#egg=manifolder&subdirectory=code'
```

If needed, the pacake can be uninstalled with `pip uninstall manifolder`.

To see a quick demo of the code, check out the [manifolder_notebook](https://github.com/babel-publishing/manifolder/blob/master/manifolder_notebook.ipynb), which calculates an underlying 3-dimensional manifold from an 8-dimensional timeseries, for both synthetic and real-world (solar wind) data.
