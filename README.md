# m_estimation_SI


## Installation

To set up an appropriate python virtual environment:

```
virtualenv env -p python3.10
source env/bin/activate
pip install -r requirements.txt
```

To install the package `regreg` you will need to run:

```
pip install git+https://github.com/regreg/regreg.git
```

Then, you can run
```
pip install .
```

To run the randomized conditional selective inference approach of Huang et al. (2025), locally install `github.com/yiling-h/PoSI-GroupLASSO`. You may need to replace `np.bool` with `bool`.

## Develop

Set up a development environment as above, but furthermore install development packages
```
pip install -r dev-requirements.txt
```
and install the package in editable mode
```
pip install -e .
```
