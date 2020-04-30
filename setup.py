# standard python setup file, see
# https://packaging.python.org/tutorials/packaging-projects/

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="manifolder",
    version="0.0.3",
    author="AV Lab (Analytics Ventures)",
    author_email="repo@avlab.com",
    description="Time-series analysis using emperical intrinstic geometry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/avlab/manifolder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
