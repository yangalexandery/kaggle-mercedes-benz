# Mercedes-Benz Kaggle Competition

Contains a solution to the Mercedes-Benz Kaggle Competition, which required contestants to predict the duration of a testing process, given 8 categorical variables and 368 binary variables. 4209 points were given for both the train and test set, contestants needed to be careful not to overfit. 

This submission ended with a 0.5777 CV R^2 score, computed using out-of-fold predictions across 5 folds.
The final model consists of an average between 5 different models:

* Gradient Boosted Tree
* Random Forest Regressor
* Ridge Regressor
* ElasticNet
* Decision Tree Regressor

A full writeup will be available to read soon on [Medium](https://medium.com/@alexyang_14414).

## Requirements

This project requires several Python packages to run:

* numpy
* pandas
* sklearn
* xgboost

Then, simply run

```
python main.py
```

## Acknowledgements

Various snippets of code were taken from Kaggle kernels. Also, special thanks to Yilun Du for suggestions.