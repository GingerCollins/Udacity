#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit


features_list = ['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments',
                 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']


with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

my_dataset = data_dict

data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=4)
selector.fit(features_train, labels_train)
print selector.scores_
print selector.get_support()


from sklearn.feature_selection import SelectPercentile
selector = SelectPercentile(percentile=25)
selector.fit(features_train, labels_train)
print selector.scores_
print selector.get_support()








