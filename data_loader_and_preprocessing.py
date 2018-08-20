import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')
import seaborn as sns
import pandas as pd
import numpy as np






class Data_Loader(object):
    
    def __init__(self, filepath, filename):
        
        self.filepath = filepath
        self.filename = filename
        self.raw_training_table = None
        self.setting_raw_training_table()
        self.enhanced_training_table = self.raw_training_table.copy()
        self.labels_per_customer = None
        
        
    def setting_raw_training_table(self):
        
        self.raw_training_table = pd.read_csv(os.path.join(self.filepath, self.filename), sep = ',')
        
        
    def getting_days_from_tmps(self, row):
    
        return row.days
    
    
    def find_next_time_of_visit(self, row):
    
        customer_table = list((self.enhanced_training_table[self.enhanced_training_table.customer == row.customer].absolute_days -
                           row.absolute_days).values)
        list_next_days = [i for i in customer_table if i > 0]
    
        if list_next_days:
            next_day = min(list_next_days)
        else:
            next_day = 'no record'
        
        return next_day
        
        
    def adding_datetime_features(self):
        
        self.enhanced_training_table['year'] = pd.to_datetime(self.raw_training_table['datetime']).dt.year
        self.enhanced_training_table['month'] = pd.to_datetime(self.raw_training_table['datetime']).dt.month
        self.enhanced_training_table['day'] = pd.to_datetime(self.raw_training_table['datetime']).dt.day
        self.enhanced_training_table['absolute_days'] = (pd.to_datetime(self.raw_training_table['datetime']) - 
                                       pd.to_datetime(self.raw_training_table['datetime'].min()))
        self.enhanced_training_table['absolute_days'] = self.enhanced_training_table['absolute_days'].apply(
            lambda x: self.getting_days_from_tmps(x))
        
        
    def removing_non_regular_customers(self):
        
        self.enhanced_training_table = self.enhanced_training_table.set_index('customer')
        nb_distinct_visits_per_custo = self.enhanced_training_table.groupby('customer')['datetime'].nunique()
        customer_to_drop = nb_distinct_visits_per_custo[nb_distinct_visits_per_custo.values <= 1].index
        self.enhanced_training_table = self.enhanced_training_table.drop(customer_to_drop)
        self.enhanced_training_table = self.enhanced_training_table.reset_index()
        
        
    
    def setting_labels_for_each_transaction(self):
        
        self.enhanced_training_table['label'] = self.enhanced_training_table.apply(lambda x: self.find_next_time_of_visit(x), axis = 1)
    
    
        
    def removing_transactions_without_future_records(self):
            
        idx_to_drop = self.enhanced_training_table[self.enhanced_training_table.label == 'no record'].index
        self.enhanced_training_table = self.enhanced_training_table.drop(idx_to_drop)

    
        
    def setting_enhanced_training_table(self):
        
        self.adding_datetime_features()
        print('datetime features were added.')
        
        self.removing_non_regular_customers()
        print('non regular customers with only 1 distinct visit were removed.')
        
        self.setting_labels_for_each_transaction()
        print('labels for each transaction were set (time-consuming).')
        
        
        self.removing_transactions_without_future_records()
        print('transactions without future records were removed.')
        
        
        return self.enhanced_training_table
    
    
    def saving_enhanced_training_table(self, save_path, save_filename):
        
        self.enhanced_training_table.to_csv(os.path.join(save_path, save_filename))
        
        
        
        
        
        
        
        
        
class DataCleaner(object):
    
    def __init__(self, training_table, min_distinct_days_customer_visit):
        
      
        self.min_distinct_days_customer_visit = min_distinct_days_customer_visit
        self.training_table = training_table
        
        print('data preprocessor loaded.')
    
    
    
    def removing_useless_columns(self):
        
        cols_to_drop = ['Unnamed: 0']
        self.training_table = self.training_table.drop(cols_to_drop, axis = 1)
        
        
    """    
    def removing_non_frequent_customers1(self):
        
        self.training_table = self.training_table.set_index(['customer'])
        nb_transactions_per_customer = self.training_table.index.value_counts()
        customer_to_drop = nb_transactions_per_customer[nb_transactions_per_customer.values < 
                                self.min_nb_transactions_per_customer].index
        self.training_table = self.training_table.drop(customer_to_drop)
        self.training_table = self.training_table.reset_index()
        
    """
    
    def removing_non_frequent_customers(self):
        
        
        nb_distinct_days_with_transactions = self.training_table.groupby('customer')['absolute_days'].nunique()
        customer_to_drop = nb_distinct_days_with_transactions[nb_distinct_days_with_transactions.values < 
                                                              self.min_distinct_days_customer_visit].index
        self.training_table = self.training_table.set_index('customer')
        self.training_table = self.training_table.drop(customer_to_drop)
        self.training_table = self.training_table.reset_index()
       
    
    
    def modify_pack_column(self, row):
    
        if type(row) == int:
            return row
        elif 'X' in row:
            return float(row.split('X')[-1])
        else:
            return float(row)
        
        
        
    def modify_discount_column(self, row):
    
        if type(row) == int or type(row) == float:
            return row
        elif 'GRA' in row:
            return float(row.split('GRA')[0])
        else:
            return float(row)
    
        
        
    def handling_nan(self):
        
        self.training_table = self.training_table.fillna(value = 'Na')
        
        # format feature
        self.training_table['format'] = self.training_table['format'].replace(to_replace = 'Na', value = 'unknown')
        
        # pack feature
        self.training_table['pack'] = self.training_table['pack'].replace(to_replace = 'Na', value = 1)
        self.training_table['pack'] = self.training_table['pack'].apply(lambda x: self.modify_pack_column(x))
        
        # discount feature
        self.training_table['discount'] = self.training_table['discount'].replace(to_replace = 'Na',  value = 0)
        self.training_table['discount'] = self.training_table['discount'].apply(lambda x: self.modify_discount_column(x))
        

        
        
    def setting_preprocessed_training_table(self):
        
        self.removing_useless_columns()
        print('useless columns were removed.')
        
        self.removing_non_frequent_customers()
        print('customer with less than ' + str(self.min_distinct_days_customer_visit) + ' distinct visits were removed.' )
        
        self.handling_nan()
        print('NaN values were handled.')
        
        return self.training_table
    
    
    
    
    
    
    
class DataAggregater(object):
   


    def __init__(self, dataset):
        
        self.dataset = dataset
        self.labels = None
        self.col_to_change = None
        self.aggregated_df = pd.DataFrame()
        
        print('The data aggregater was initialized.')
    
    
    
    def original_dataset_to_aggregated_dummies(self):
        
        self.dataset = pd.get_dummies(self.dataset, columns = ['category', 'discount', 'pack'])
        self.col_to_change = [i for i in list(self.dataset.columns) if ('_' in i) and 
                 (i != 'absolute_days') and (i != 'original_format')]
        for col in self.col_to_change:
            self.dataset[col] = self.dataset[col] * self.dataset['quantity']
        
        
        
    def extract_label_and_remove_last_transaction(self):
                
        idx = self.dataset.groupby(['customer'])['absolute_days'].transform(max) == self.dataset['absolute_days']
        self.label = self.dataset[idx].set_index('customer').sort_index()
        # now we get rid off these transactions, they should not interact with the dataset. 
        rows_to_drop = [ i for i in range(idx.shape[0]) if idx[i] == True]
        self.dataset = self.dataset.drop(rows_to_drop)
        self.dataset['between9_10'] = self.dataset.datetime.apply(lambda x: int(x[11:13]) >= 9 and int(x[11:13]) < 10  )
        self.dataset['between11_13'] = self.dataset.datetime.apply(lambda x: int(x[11:13]) >= 11 and int(x[11:13]) < 13  )
        self.dataset['above20'] = self.dataset.datetime.apply(lambda x: int(x[11:13]) >= 20  )
        
        
    
    def setting_aggregated_features(self):
        
        
        self.aggregated_df['customer'] = self.dataset.customer.unique()
        self.aggregated_df = self.aggregated_df.set_index('customer')
        self.aggregated_df['labels'] = self.label.groupby('customer')['label'].max()
        self.aggregated_df['total_quantity'] = self.dataset.groupby('customer')['quantity'].agg('sum', axis = 1)
        self.aggregated_df['total_transactions'] = self.dataset.groupby('customer')['customer'].count()
        self.aggregated_df['nb_distinct_days_with_transactions'] = self.dataset.groupby('customer')['absolute_days'].nunique()
        self.aggregated_df['lag1_time_between_visits'] = self.dataset.groupby('customer')['label'].unique().agg(lambda x: x[-1])
        self.aggregated_df['lag2_time_between_visits'] = self.dataset.groupby('customer')['label'].unique().agg(lambda x: x[-2])
        #self.aggregated_df['lag3_time_between_visits'] = self.dataset.groupby('customer')['label'].unique().agg(lambda x: x[-3])
        
        for col in self.col_to_change:
            self.aggregated_df[col] = self.dataset.groupby('customer')[col].agg('sum')

        for operation in ['min', 'max', 'mean', 'median', 'std']:
            self.aggregated_df[operation + '_time_between_visits'] = self.dataset.groupby('customer')['label'].agg(operation)
        
        self.aggregated_df['between9_10'] = self.dataset.groupby('customer')['between9_10'].sum()
        self.aggregated_df['between11_13'] = self.dataset.groupby('customer')['between11_13'].sum()
        self.aggregated_df['above20'] = self.dataset.groupby('customer')['above20'].sum()
            
    
    def create_aggregated_dataframe(self):
        
        self.original_dataset_to_aggregated_dummies()
        
        self.extract_label_and_remove_last_transaction()
        
        self.setting_aggregated_features()
        
        print('The aggregated dataframe per customer and their respective labels (last transaction of a given customer) were computed.')
        
        return self.aggregated_df, self.labels
    
    
    
    
    
        
