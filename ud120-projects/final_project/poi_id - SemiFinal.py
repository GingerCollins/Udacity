#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
# check out data
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print '# of data points:', len(data_dict)
pois = 0
non_pois = 0
for d in data_dict:
    if data_dict[d]['poi']:
        pois += 1
    else:
        non_pois += 1
print '{0:.0%}'.format(pois / float(len(data_dict))), ' POI / ', '{0:.0%}'.format(non_pois / float(len(data_dict))), ' non-POI'

# learned code from stackoverflow
unique_features = set(feature
                      for row_dict in data_dict.values()
                      for feature in row_dict.keys())
print 'total # of features:', len(unique_features)

# get number of nan from data_dict
features_with_nan = {}
for d in data_dict:
    for c in data_dict[d]:
        if data_dict[d][c] == 'NaN':
            if c not in features_with_nan:
                features_with_nan[c] = 1
            else:
                features_with_nan[c] += 1
print 'features and # of nan values'
for r in features_with_nan:
    print r, features_with_nan[r]
print 'total # of features with NaN:', len(features_with_nan)
# poi is the only feature with no NaN in the values

# decide what features to use
test_features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options',
                      'director_fees', 'deferral_payments', 'deferred_income', 'expenses', 'from_this_person_to_poi',
                      'loan_advances', 'long_term_incentive', 'other', 'restricted_stock', 'restricted_stock_deferred',
                      'shared_receipt_with_poi', 'to_messages', 'total_payments']
# was picking the features with some of the most NaN in data, from_this_person_to_poi, loan_advances, 'shared_receipt_with_poi'

test_features_list = ['poi', 'director_fees', 'total_payments', 'to_messages', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options',
                      'deferred_income', 'expenses', 'long_term_incentive', 'other', 'restricted_stock']
my_dataset = data_dict
data = featureFormat(my_dataset, test_features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(features)
# features = scaler.transform(features)

from sklearn.feature_selection import SelectKBest
select = SelectKBest(k=3)
select.fit(features, labels)
print select.get_support()
print select.get_support(indices=True)

# Select K Best advises that salary, bonus, total stock value and exercised stock options are 4 variables we need to use
# Select K Best advises that salary, bonus and total stock value are 3 variables we need to use
# I tested the above without and with a MinMaxScaler to make sure there was no difference


from sklearn.feature_selection import SelectPercentile
select = SelectPercentile(percentile=15)
select.fit(features, labels)
print select.get_support()
print select.get_support(indices=True)

# Select Percentile select the same variables as Select K Best when percentile is set to 20 without and with MinMaxScaler

# features selected was salary, bonus, total_stock_value and exercised_stock_options percentile=20
# features selected was salary, bonus and total_stock_value percentile=15
# since exercised_stock_options is part of total_stock_value I think this feature should not be used.


### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'total_payments', 'total_stock_value'] # You will need to use more features
#features_list = ['poi', 'salary', 'bonus']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
import matplotlib.pyplot as plt


def plot(data):
    for d in data:
        if data[d]['total_payments'] != 'NaN' and data[d]['total_stock_value'] != 'NaN':
            plt.scatter(data[d]['total_payments'], data[d]['total_stock_value'])
    plt.xlabel('Total Payments')
    plt.ylabel('Total Stock Value')

    #plt.show()


plot(data_dict)

highest_salary = 309886585
second_highest_salary = 103559793
outlier_highest_salary = ''
second_outlier_highest_salary = ''
for d in data_dict:
    if data_dict[d]['total_payments'] != 'NaN':
        # if data_dict[d]['total_payments'] > highest_salary:
        #     highest_salary = data_dict[d]['total_payments']
        if highest_salary > data_dict[d]['total_payments'] > second_highest_salary:
            second_highest_salary = data_dict[d]['total_payments']
        if data_dict[d]['total_payments'] == highest_salary:
            outlier_highest_salary = d
        if data_dict[d]['total_payments'] == second_highest_salary:
            second_outlier_highest_salary = d

# print highest_salary
print 'outlier with highest salary:', outlier_highest_salary
# print second_highest_salary
print 'outlier with second highest salary:', second_outlier_highest_salary

data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
plot(data_dict)

# need to remove NaN values could use zscore to remove other outliers
datapoint_to_remove = []
for dd in data_dict:
    if data_dict[dd]['total_payments'] == 'NaN' and data_dict[dd]['total_stock_value'] == 'NaN':
        datapoint_to_remove.append(dd)
    if data_dict[dd]['total_payments'] == 0 and data_dict[dd]['total_stock_value'] == 0:
        if datapoint_to_remove.append(dd) not in datapoint_to_remove:
            datapoint_to_remove.append(dd)

# for l in datapoint_to_remove:
#     data_dict.pop(l, 0)
# plot(data_dict)
# print datapoint_to_remove

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# from using the feature selection modules from sklearn I can create a new feature using salary and bonus and have the total stock value as second feature
new_features_list = ['poi', 'salary', 'bonus', 'total_stock_value']
new_feature_my_dataset = data_dict
new_data = featureFormat(new_feature_my_dataset, new_features_list, sort_keys=True)
new_labels, new_features = targetFeatureSplit(new_data)

import pandas as pd
df = pd.DataFrame(new_data, columns=new_features_list)

# scale data
df_to_scale = pd.DataFrame(df[['salary', 'bonus', 'total_stock_value']])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(df_to_scale)
df_scaled = pd.DataFrame(scaled, columns=['salary', 'bonus', 'total_stock_value'])
df_to_combine = pd.DataFrame(df_scaled[['salary', 'bonus']])

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
p = pca.fit_transform(df_to_combine)
df_combined = pd.DataFrame(data=p, columns=['Principal'])
df_label = pd.DataFrame(df[['poi']])
df_stock = pd.DataFrame(df_scaled[['total_stock_value']])
df_features = pd.concat([df_combined, df_stock], axis=1)

### Extract features and labels from dataset for local testing
# this is my manual pick for final testing
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# using data with new feature
import numpy as np
new_labels_array = np.array(df_label)
new_feature_array = np.array(df_features)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
new_clf = GaussianNB()
new_clf.fit(new_feature_array, new_labels_array.ravel())
pred = new_clf.predict(new_feature_array)
print 'GaussianNB new feature'
print 'new feature accuracy score:', accuracy_score(new_labels_array.ravel(), pred)
print 'new feature precision scoore:', precision_score(new_labels_array, pred)
print 'new feature recall score:', recall_score(new_labels_array, pred)
print ''


# using manaully selected features
clf1 = GaussianNB()
clf1.fit(features, labels)
pred = clf1.predict(features)
print 'GaussianNB manual'
print 'accuracy score:', accuracy_score(labels, pred)
print 'precision scoore:', precision_score(labels, pred)
print 'recall score:', recall_score(labels, pred)
print ''


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features, labels)
pred = clf.predict(features)
print 'Decision Tree Classifer'
print 'accuracy:', accuracy_score(labels, pred)
print 'precision:', precision_score(labels, pred)
print 'recall:', recall_score(labels, pred)
print ''


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# new feature
from sklearn.model_selection import train_test_split
new_features_train, new_features_test, new_labels_train, new_labels_test = \
    train_test_split(new_feature_array, new_labels_array.ravel(), test_size=0.3, random_state=42)

new_clf = GaussianNB()
new_clf.fit(new_features_train, new_labels_train)
pred = new_clf.predict(new_features_test)
print 'GaussianNB final new feature'
print 'final new feature accuracy score:', accuracy_score(new_labels_test, pred)
print 'final new feature precision scoore:', precision_score(new_labels_test, pred)
print 'final new feature recall score:', recall_score(new_labels_test, pred)
print 'n'

# manually selected features
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# clf = GaussianNB()
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# print 'final accuracy score:', accuracy_score(labels_test, pred)
# print 'final precision scoore:', precision_score(labels_test, pred)
# print 'final recall score:', recall_score(labels_test, pred)
#

# clf = DecisionTreeClassifier()
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# print 'Decision Tree Classifer'
# print 'final accuracy:', accuracy_score(labels_test, pred)
# print 'final precision:', precision_score(labels_test, pred)
# print 'recall:', recall_score(labels_test, pred)

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=2)
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# print pred
# print 'KNeighborsClassifier'
# print 'accuracy:', accuracy_score(labels_test, pred)
# print 'precision:', precision_score(labels_test, pred)
# print 'recall:', recall_score(labels_test, pred)


# things to try
# GridSearchCV (scoring = f1) use sss=StratifiedShuffleSplit(labels) search=GridSearchCV(...., cv=sss)
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(SVC(), cv=2)
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# print 'GridSearchCV'
# print 'accuracy:', accuracy_score(labels_test, pred)
# print 'precision:', precision_score(labels_test, pred)
# print 'recall:', recall_score(labels_test, pred)

# clf = SVC(kernel="linear")
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# print 'accuracy:', accuracy_score(labels_test, pred)
# print 'precision:', precision_score(labels_test, pred)
# print 'recall:', recall_score(labels_test, pred)
#
# from sklearn.cross_validation import StratifiedShuffleSplit
#
# labels, features = targetFeatureSplit(data)
# sss = StratifiedShuffleSplit(labels, 1000, random_state=42)
#
# f_train = []
# f_test = []
# t_train = []
# t_test = []
# for train_idx, test_idx in sss:
#     for ii in train_idx:
#         f_train.append(features[ii])
#         t_train.append(labels[ii])
#     for jj in test_idx:
#         f_test.append(features[jj])
#         t_test.append(labels[jj])
#
#
# clf = DecisionTreeClassifier()
# clf.fit(f_train, t_train)
# pred = clf.predict(f_test)
# print 'Decision Tree Classifer'
# print 'accuracy:', accuracy_score(t_test, pred)
# print 'precision:', precision_score(t_test, pred)
# print 'recall:', recall_score(t_test, pred)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)