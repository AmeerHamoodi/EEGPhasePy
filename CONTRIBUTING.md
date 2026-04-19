# Contributing to EEGPhasePy

## Getting started

Fork the repository on GitHub and clone your fork, branching off the latest `master`.

```bash
git clone https://github.com/<your-username>/EEGPhasePy.git
cd EEGPhasePy
git checkout master
git checkout -b my-feature
```

## Setting up your environment

Create and activate a Python virtual environment, then install the package dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

To also work on the documentation, install the docs dependencies:

```bash
pip install -r docs/requirements.txt
```

## Running the tests

```bash
pytest
```

## Building the documentation

```bash
cd docs
make html
```

The built HTML will be at `docs/build/html/index.html`.

## Submitting a pull request

1. Make your changes on your feature branch.
2. Add or update tests to cover your changes.
3. Add or update the relevant documentation.
4. Open a pull request against `master` on the main repository.
