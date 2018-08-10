#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'total_stock_value'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys=True)
#labels, features = targetFeatureSplit(data)
import pandas as pd
df = pd.DataFrame(data, columns=features_list)
# print df

# scale data
df_to_scale = pd.DataFrame(df[['salary', 'bonus', 'total_stock_value']])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(df_to_scale)
df_scaled = pd.DataFrame(scaled, columns=['salary', 'bonus', 'total_stock_value'])
# print scaled
df_to_combine = pd.DataFrame(df_scaled[['salary', 'bonus']])

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
p = pca.fit_transform(df_to_combine)
df_combined = pd.DataFrame(data=p, columns=['Principal'])

df_label = pd.DataFrame(df[['poi']])
df_stock = pd.DataFrame(df_scaled[['total_stock_value']])
df_feature = pd.concat([df_combined, df_stock], axis=1)
# print df_feature






### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
import numpy
l = df_label.values
f = df_feature.values
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
clf = GaussianNB()
# clf.fit(f, l.ravel())
# pred = clf.predict(f)
# print accuracy_score(l, pred)
#0.8604651162790697

# clf.fit(features, labels)
# pred = clf.predict(features)
# print accuracy_score(labels, pred)
#0.8527131782945736


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print accuracy_score(labels_test, pred)
#0.8717948717948718

features_list = ['poi', 'total_payments', 'total_stock_value'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
my_dataset = data_dict
#data_dict.pop('TOTAL', 0)
#data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(scaled, labels, test_size=0.3, random_state=42)
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print accuracy_score(labels_test, pred)
#0.9302325581395349

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)