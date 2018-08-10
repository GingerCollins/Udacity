#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
# added modules
import matplotlib.pyplot as plt


# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'director_fees', 'total_payments', 'total_stock_value',
                 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
# print data_dict

# Task 2: Remove outliers

people = 0
features = 0
pois = 0
NaN_features = {}
count = 0
for p in data_dict:
    # print p
    people += 1
    for k in data_dict[p]:
        features += 1
        if k == 'poi':
            if data_dict[p][k]:
                pois += 1
# print '# of people/data points in dataset:', people
# print '# of POIs in the dataset:', pois
# print 'allocation across classes POI/non-POI:', "{:.0%}".format((pois/float(people))),  "{:.0%}".format((people-pois)/float(people))
# print '# of dataset features per person:', features / people
# print '# of of features used:', len(features_list)


# features with missing values NaN in dataset
dict1 = {}
dict2 = {}
for d in data_dict:
    for k, v in data_dict[d].iteritems():
        # print k, v
        if v == 'NaN':
            if data_dict[d]['poi']:
                if k not in dict2:
                    dict2[k] = 1
                else:
                    dict2[k] = dict2[k] + 1

            if k not in dict1:
                dict1[k] = 1
            else:
                dict1[k] = dict1[k] + 1
# for l in dict1:
    # print l, dict1[l]
# print "# features with NaN in the dataset:", len(dict1)
# print 'features with no NaN in the dataset:'
# for d in data_dict:
#     for k in data_dict[d]:
#         if k not in dict1:
#             print '----', k
#     break

# print "# features with NaN in the poi dataset:", len(dict2)
print 'features with no NaN in the poi dataset:'
for d in data_dict:
    for k in data_dict[d]:
        if k not in dict2:
            print '----', k
    break
dict1 = {}
dict2 = {}

# features with missing values NaN in features used
for d in data_dict:
    for k, v in data_dict[d].iteritems():
        if v == 'NaN':
            if data_dict[d]['poi']:
                if k not in dict2 and k in features_list:
                    dict2[k] = 1
                elif k in dict2 and k in features_list:
                    dict2[k] = dict2[k] + 1
            if k not in dict1 and k in features_list:
                dict1[k] = 1
            elif k in dict1 and k in features_list:
                dict1[k] = dict1[k] + 1
print 'features with no NaN in feature list'
for f in features_list:
    if f not in dict1:
        print "-----", f

# print '# of features with NaN in features list:', len(dict1)
# print '# of features with NaN in poi features list:', len(dict2)
print 'features with no NaN in poi features list'
for p in features_list:
    if p not in dict2:
        print '-----', p


# number of data points that have no NaN in all feature list
people_with_no_nan = 0
for d in data_dict:
    if data_dict[d]['poi'] != 'NaN':
        if data_dict[d]['total_payments'] != 'NaN':
            if data_dict[d]['total_stock_value'] != 'NaN':
                people_with_no_nan += 1
print '# of people with no NaN in feature list:', people_with_no_nan


# plotting for outliers for total_payments and total_stock_value
htp = 0
person_highest_htp = ''
highest_htp = 309886585
second_highest_htop = ''
for p in data_dict:
    if data_dict[p]['total_payments'] != 'NaN' and data_dict[p]['total_stock_value'] != 'NaN':
        if data_dict[p]['total_payments'] > htp:
            htp = data_dict[p]['total_payments']
            person_highest_htp = p
            if data_dict[p]['total_payments'] < highest_htp:
                second_highest_htop = p
        plt.scatter(data_dict[p]['total_payments'], data_dict[p]['total_stock_value'])

plt.xlabel("total_payments")
plt.ylabel("total_stock_value")
plt.show()

# print 'person with highest total payments:', person_highest_htp
# print htp
# print 'person with second highest total payments:', second_highest_htop

# removed outliers: TOTAL, other outlier is actually not an outlier but for SKILLING JEFFREY K
# removed company from the data since this is not a person (from PDF)
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

# looking for outliers in poi data only in total_payments and total_stock_value
htp = 0
person_htp = 0
datapoints_with_no_salary_and_stock = []
for p in data_dict:
    if data_dict[p]['poi']:
        if data_dict[p]['total_payments'] > htp:
            htp = data_dict[p]['total_payments']
            person_htp = p
        plt.scatter(data_dict[p]['total_payments'], data_dict[p]['total_stock_value'])
    if data_dict[p]['salary'] == 'NaN' and data_dict[p]['total_stock_value'] == 'NaN':
        datapoints_with_no_salary_and_stock.append(p)

plt.xlabel("total_payments")
plt.ylabel("total_stock_value")
plt.show()
print '# with no salary and total_stock_value:', len(datapoints_with_no_salary_and_stock)
for l in datapoints_with_no_salary_and_stock:
    print '----', l
# do we need to remove the 15 people besides THE TRAVEL AGENCY IN THE PARK? since they have no salary or stock
# print 'poi with highest total payments:', person_htp
# print htp


# Task 3: Create new feature(s)

# Select K Best
features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options',
                 'director_fees', 'deferral_payments', 'deferred_income', 'expenses', 'from_this_person_to_poi',
                 'loan_advances', 'long_term_incentive', 'other', 'restricted_stock', 'restricted_stock_deferred',
                 'shared_receipt_with_poi', 'to_messages', 'total_payments']

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(features)
# features = scaler.transform(features)


from sklearn.feature_selection import SelectKBest
select = SelectKBest(k=3)
select.fit(features, labels)
print select.get_support()
print select.get_support(indices=True)
select.transform(features)

# Select K Best advises that salary, bonus, total stock value and exercised stock options are 4 variables we need to use
# Select K Best advises that salary, bonus and total stock value are 3 variables we need to use
# I tested the above without and with a MinMaxScaler to make sure there was no difference

# Percentile

from sklearn.feature_selection import SelectPercentile
select = SelectPercentile(percentile=15)
select.fit(features, labels)
print select.get_support()
print select.get_support(indices=True)
select.transform(features)
# Select Percentile select the same variables as Select K Best when percentile is set to 20 without and with MinMaxScaler

# features selected was salary, bonus, total_stock_value and exercised_stock_options percentile=20
# features selected was salary, bonus and total_stock_value percentile=15
# since exercised_stock_options is part of total_stock_value I think this feature should not be used.
# from using the feature selection modules from sklearn I can create a new feature using salary and bonus and have the total stock value as second feature


# need to create new feature using salary and bonus and compare to total_stock_value
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
features_list = ['poi', 'total_stock_value', 'salary', 'bonus']

# featureFormat removed some data that has NaN in all values
data = featureFormat(data_dict, features_list)
# labels, feature = targetFeatureSplit(data)
# dictionary to dataframe
df = pd.DataFrame(data, columns=['poi', 'stocks', 'salary', 'bonus'])
# print df

df_labels = df[['poi']]
df_feature = df[['stocks', 'salary', 'bonus']]
#print feature

scaled = StandardScaler().fit_transform(df_feature)
df_scaled = pd.DataFrame(scaled, columns=['stocks', 'salary', 'bonus'])
df_stocks = df_scaled[['stocks']]
df_to_combine = df_scaled[['salary', 'bonus']]

# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
pca = PCA(n_components=1)
principalComponents = pca.fit_transform(df_to_combine)
principalDF = pd.DataFrame(data=principalComponents, columns=['principal'])


#print principalDF

df1 = pd.concat([df_labels, principalDF, df_stocks], axis=1)

# when looking at data for all features manually I would select total payments and total stock values
# since these 2 features are the total value of poi



features_list = ['poi', 'total_payments', 'total_stock_value']

my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
clf = GaussianNB()
clf.fit(features, labels)
pred = clf.predict(features)
print 'precision:', precision_score(labels, pred)
print 'recall:', recall_score(labels, pred)



# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_list = ['poi', 'total_payments', 'total_stock_value']

my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print 'GaussianNB'
print 'precision:', precision_score(labels_test, pred)
print 'recall:', recall_score(labels_test, pred)

data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print 'Decision Tree Classifer'
print 'precision:', precision_score(labels_test, pred)
print 'recall:', recall_score(labels_test, pred)


data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.svm import SVC
clf = SVC()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print 'SVC'
print 'accuracy:', clf.score(labels_test, pred)
print 'precision:', precision_score(labels_test, pred)
print 'recall:', recall_score(labels_test, pred)



# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)