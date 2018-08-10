#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

features_list = ['poi', 'total_payments', 'total_stock_value']
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
my_dataset = data_dict

data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
clf = GaussianNB()
clf.fit(features, labels)
pred = clf.predict(features)

print 'accuracy:', accuracy_score(labels, pred)
print 'precision:', precision_score(labels, pred)
print 'recall:', recall_score(labels, pred)
print 'f1:', f1_score(labels, pred)
# salary 0.2
# bonus .34146
# total payments 0.856
# total stock value 0.857
# total payments and stock value 0.874

print 'train test split'
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print 'accuracy:', accuracy_score(labels_test, pred)
print 'precision:', precision_score(labels_test, pred)
print 'recall:', recall_score(labels_test, pred)
print 'f1:', f1_score(labels_test, pred)
# salary 0.34482758
# bonus 0.56
# total payments 0.78947
# total stock value 0.8947368
# total payments and stock value 0.93023255, precision=0.75, recall=0.6

print 'standard scaler'
from sklearn.preprocessing import StandardScaler
labels, features = targetFeatureSplit(data)

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

features_train, features_test, labels_train, labels_test = \
    train_test_split(scaled, labels, test_size=0.3, random_state=42)

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print 'accuracy:', accuracy_score(labels_test, pred)
print 'precision:', precision_score(labels_test, pred)
print 'recall:', recall_score(labels_test, pred)
print 'f1:', f1_score(labels_test, pred)

#'total_payments', 'total_stock_value'
# accuracy: 0.9302325581395349
# precision: 0.75
# recall: 0.6
# f1: 0.6666666666666665
# there is no change in data when using standard scaler

print 'min max scaler'
from sklearn.preprocessing import MinMaxScaler
labels, features = targetFeatureSplit(data)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

features_train, features_test, labels_train, labels_test = \
    train_test_split(scaled, labels, test_size=0.3, random_state=42)

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print 'accuracy:', accuracy_score(labels_test, pred)
print 'precision:', precision_score(labels_test, pred)
print 'recall:', recall_score(labels_test, pred)
print 'f1:', f1_score(labels_test, pred)
# no change between StandardScaler and MinMaxScaler

#'exercised_stock_options', 'total_stock_value':
# accuracy: 0.8947368421052632
# precision: 1.0
# recall: 0.3333333333333333
# f1: 0.5