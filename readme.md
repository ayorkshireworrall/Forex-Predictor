Forex Prediction
=============
Table of contents
-------
* [Project Overview](#project-overview)
* [Getting Started](#getting-started)

Project Overview
-------
This is a project designed to make predictions based on forex data. It will make use of neural networks to categorise a data pattern as either an opportunity to buy a currency, or sell. The inputs will be a series of open-high-low-close (ohlc) data points, the time width and the number of which will be configurable in order to try and find the best patterns. The outputs will be a classification of buy or sell within a certain timeframe, again this will be configurable to find the best fit.

The project will not provide particularly accurate predictions as the data is generally random. It is known that certain traders use patterns to gain an indication of whether to buy/sell and form a strategy that gives them a very slight edge that pays off over the long run. This is the intention of the project, and success will be defined as gaining average trade profit that exceeds the buy-spread set by common brokers.

Data provided for training, test and validation is downloaded from https://www.histdata.com/ The ascii format will create csv files which are merged and their datetimes reformatted to adhere to more common conventions.


Getting Started
-------
Clone the repo:
```
git clone https://gitlab.com/aworrall2512/forex-predictor.git
```

Using the library virtual env (version 20.0.31 at the time of writing), create a new virtual environment
```
pip install virtualenv
virtualenv .venv
```
Add the root path to the generated file `.venv/Lib/site-packages/_virtual_env.py`, below the imports add

```python
sys.path.append('path\\to\\project\\ForexPredictor')
```
This will allow the standard relative imports to work when running any scripts from the root directory (useful when working as a VS Code project)
