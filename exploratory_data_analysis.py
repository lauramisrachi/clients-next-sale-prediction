import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')
import seaborn as sns
import pandas as pd
import numpy as np
import pandas as pd
import os




class EDA(object):
    
    
    def __init__(self, training_table, continuous_features, categorical_features, id_features):
        
        self.training_table = training_table
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.id_features = id_features
        
        print('EDA plotter initialized')
    
    def add_new_feature(self, feature, category):
        
        if category == 'continuous':
            continuous_features.append(feature)
        elif category == 'categorical':
            categorical_features.append(feature)
        else:
            id_features.append(feature)
        
    def plot_categorical_instance(self, feature):
        
        self.training_table[feature].value_counts(normalize = True, dropna = False).plot(kind = 'bar')
        plt.title(feature + ' feature instances across our dataset.')
        plt.show()
        
    def plot_continuous_instance(self, feature):
        
        self.training_table[feature].hist(bins = 50)
        plt.title(feature + ' feature histogram distribution across our dataset.')
        plt.show()
    
    def plot_feature_instances(self, feature):
        
        if feature in self.categorical_features:
            self.plot_categorical_instance(feature)
        else:
            self.plot_continuous_instance(feature)
        

        
        
        
        
        
class Plotter(object):
    
    def __init__(self, training_table):
        
        self.training_table = training_table

        
    def autolabel(self, rects, ax):
        
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')
            
    def plot_bar(self, df):
        
        N = len(df.index)
        means = df['mean']
        std = df['std']

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, means, width, color='r', yerr=std)

        # add some text for labels, title and axes ticks
        ax.set_title(df.index.name)
        ax.set_xticks(ind)
        ax.set_xticklabels(df.index)
        
        self.autolabel(rects1, ax)
        
        plt.show()
        
    def plot_correlation_matrix(self, size = 9):
    
        corr = self.training_table.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
        
        plt.xticks(range(len(corr.columns)), corr.columns);
        plt.yticks(range(len(corr.columns)), corr.columns);
        plt.title('Correlation matrix')
        
        plt.show()
    
    def plot_label_distribution(self):
        
        self.training_table['label'].hist(bins = 50)
        plt.title('Label repartition')
        plt.show()
        