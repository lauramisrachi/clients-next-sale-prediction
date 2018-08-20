from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')
import seaborn as sns
import pandas as pd
import numpy as np


class Model(object):
    
    def __init__(self, model, X, y, n_splits, test_size, score, tuned_parameters):
        
        
        self.model = model
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.n_splits = n_splits
        self.test_size = test_size
        self.score = score
        self.tuned_parameters = tuned_parameters
        self.best_params = None
        self.results = None
        self.importances = None
        
        
        print('Modeler initialized.')
        
    
    def preprocess(self):
        
        
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
             
    def make_train_test_split(self):
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                test_size = self.test_size, random_state = 0)
        
        print('train-test split done, with test size of ' + str(self.test_size))
        
        if self.model != RandomForestRegressor:
            self.preprocess()
            print('Standard scaling of the data was done.')
        
    
    def cv_for_hyperparameter_search(self):
        
        clf_GridSearch = GridSearchCV(self.model(), self.tuned_parameters, scoring = self.score, cv = self.n_splits)
        clf_GridSearch.fit(self.X_train, self.y_train)
        self.best_params = clf_GridSearch.best_params_
        self.results = clf_GridSearch.cv_results_
        
        print('Crossvalidation for hyperparameter optimization was done.')
        
        
    def plot_crossvalidation_result(self):
        
        f, (ax1, ax2) = plt.subplots(1,2, sharey=True)
        

        x = range(len(self.results['params']))
        ax1.errorbar(x, self.results['mean_test_score'], yerr = self.results['std_test_score'], fmt = 'bo', label = 'test cv score')
        # Setting the labels for the x-axis (gridsearch combination)
        x_ticks_labels = [str(tuple(self.results['params'][i].values())) for i in range(len(self.results['params']))]
        # Set number of ticks for x-axis
        ax1.set_xticks(x);
        # Set ticks labels for x-axis
        ax1.set_xticklabels(x_ticks_labels, rotation=90, fontsize=8);


        ax2.errorbar(x, self.results['mean_train_score'], yerr = self.results['std_train_score'], fmt = 'ro', label = 'train cv score')
        
        # Setting the labels for the x-axis (gridsearch combination)
        x_ticks_labels = [str(tuple(self.results['params'][i].values())) for i in range(len(self.results['params']))]
        # Set number of ticks for x-axis
        ax2.set_xticks(x);
        # Set ticks labels for x-axis
        ax2.set_xticklabels(x_ticks_labels, rotation=90, fontsize=8);


        ax1.legend(bbox_to_anchor=(-0.5, 0.5), loc=4, borderaxespad=0., fontsize = 10);
        ax1.set_ylabel('Negative mean squared error');

        ax2.legend(bbox_to_anchor=(-1.69, 0.6), loc=4, borderaxespad=0., fontsize = 10);

        plt.title('Hyperparameter selection with 5-fold CV : (max_depth, min_samples_leaf, n_estimators)')
        plt.title('Hyperparameter selection with' + str(self.n_splits) + '- fold CV  ' + str(list(self.tuned_parameters.keys())) )
     
    
    def fit_and_predict_best_model(self):
        
        
        clf = self.model(**self.best_params)
        clf.fit(self.X_train, self.y_train)
        self.y_pred = clf.predict(self.X_test)
        
        print('best model was fitted to the test set and predictions were computed.')
        
        
    def get_test_score(self):
        
        rmse_train = np.sqrt(abs(self.results['mean_train_score'].mean()))
        rmse_test = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        mae_test = mean_absolute_error(self.y_test, self.y_pred)
        r2_test = r2_score(self.y_test, self.y_pred)
        
        print('root mean squared error in train : ', rmse_train)
        print('root mean squared error in test :', rmse_test)
        print('mean absolute error in test : ', mae_test)
        print('r2 score in test :', r2_test)
        
    
    def plot_feature_importance(self):
        
        if self.model != RandomForestRegressor:
            raise ValueError('feature importance can only be computed with RandomForestRegressor as the chosen model.')
            
        clf = self.model(**self.best_params)
        clf.fit(self.X_train, self.y_train)
        self.importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis = 0)
        indices = np.argsort(self.importances)[::-1]
        
        plt.figure()
        plt.title('Feature importances')
        plt.bar(range(self.X_train.shape[1]), self.importances[indices], color = 'r', yerr = std[indices], 
               align = 'center')
        plt.xticks(range(self.X_train.shape[1]), self.X_train.columns[indices], rotation = 90)
        plt.xlim([-1, self.X_train.shape[1]])
        plt.show()
        
        
    def plot_error_analysis(self):
    
        
        y_error = np.abs(self.y_pred - self.y_test)
        plt.plot(self.y_test.values, y_error.values, '.')
        plt.xlabel('true label')
        plt.ylabel('absolute error between true and predicted label')
        plt.title('Error analysis')
    
    