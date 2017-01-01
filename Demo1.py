# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:11:27 2016

Kaggle Titanic

@author: Min
"""
# IMPORTS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

# SET STYLE: WHITE GRID
# STYLE(DARKGRID, WHITEGRID, DARK, WHITE, TICKS)
sns.set_style('whitegrid')

# GET CSV FILE
train = pd.read_csv('train.csv', dtype = {'Age': np.float64})
test = pd.read_csv('test.csv', dtype = {'Age': np.float64})

# PREVIEW DATA IN IPYTHON CONSOLE
print(train.head())
# .head(): RETURNS FIRST N(DEFAUT N=5) ROWS OF DATAFRAME
# PREVIEW DATA INFO 
# .info(): CONCISE SUMMARY OF A DATAFRAME
print(train.info())
print('----------------------------------')
print(test.info())

# DROP UNNECESSART COLUMNS
train = train.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)
test = test.drop(['Name', 'Ticket'], axis = 1)

# EMBARKED

# TEST IF 'EMBARKED' HAS NULL VALUE -train[train['Embarked'].isnull()]
# FILL THE NULL VALUE WITH THE MOST OCCURERED VALUE 'S'
# .fillna(): FILL NA/NaN WITH SPECIFIC VALUE
train['Embarked'] = train['Embarked'].fillna('S')

# PLOT
# .factorlot(): X AXIS - EMBARKED, Y AXIS - SURVIVED, DATA FROM TRAIN
# THE VERTICIAL LINES IN THE PLOTS ARE CALLED ERRORBARS!(CONTROL BY OPTION: CI)
# SHOW THE RELATIONSHIP BETWEEN 'EMBARKED' AND 'SURVIVED'
sns.factorplot('Embarked', 'Survived', data = train, size = 4, aspect = 3)

# CREATE 3 AXISES
fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (15, 5))
# COUNT PASSENGERS EMBARKED PLACE
# sns.factorplot('Embarked', data = train, kind = 'count', order = ['S', 'C', 'Q'], ax = axis1)
sns.countplot(x = 'Embarked', data = train, ax = axis1)
# GROUP BY EMBARKED, COUNT PASSENGERS SURVIVED OR NOT
# sns.factorplot('Survived', hue = 'Embarked', data = train, kind = 'count', order = [1, 0], ax = axis2)
sns.countplot(x = 'Survived', hue = 'Embarked', data = train, order = [1, 0], ax = axis2)
# GROUP BY EMBARKED, AND GET THE MEAN FOR SURVIVED PASSENGERS FOR EACH VALUE IN EMBARKED
embark_perc = train[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()
sns.barplot(x = 'Embarked', y = 'Survived', data = embark_perc, order = ['S', 'C', 'Q'], ax = axis3)

#下面就不是我写的了
# Fare

# only for test, since there is a missing "Fare" values
test["Fare"].fillna(test["Fare"].median(), inplace=True)

# convert from float to int
train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)

# get fare for survived & didn't survive passengers 
fare_not_survived = train["Fare"][train["Survived"] == 0]
fare_survived     = train["Fare"][train["Survived"] == 1]

# get average and std for fare of survived/not survived passengers
avgerage_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare      = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])

# plot
train['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))

avgerage_fare.index.names = std_fare.index.names = ["Survived"]
avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)

# Age 

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# axis3.set_title('Original Age values - Test')
# axis4.set_title('New Age values - Test')

# get average, std, and number of NaN values in train
average_age_titanic   = train["Age"].mean()
std_age_titanic       = train["Age"].std()
count_nan_age_titanic = train["Age"].isnull().sum()

# get average, std, and number of NaN values in test
average_age_test   = test["Age"].mean()
std_age_test       = test["Age"].std()
count_nan_age_test = test["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# plot original Age values
# NOTE: drop all null values, and convert to int
train['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
# test['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# fill NaN values in Age column with random values generated
train["Age"][np.isnan(train["Age"])] = rand_1
test["Age"][np.isnan(test["Age"])] = rand_2

# convert from float to int
train['Age'] = train['Age'].astype(int)
test['Age']    = test['Age'].astype(int)
        
# plot new Age Values
train['Age'].hist(bins=70, ax=axis2)
# test['Age'].hist(bins=70, ax=axis4)

# .... continue with plot Age column

# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()

# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)

# Cabin
# It has a lot of NaN values, so it won't cause a remarkable impact on prediction
train.drop("Cabin",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)

# Family

# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
train['Family'] =  train["Parch"] + train["SibSp"]
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'] =  test["Parch"] + test["SibSp"]
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0

# drop Parch & SibSp
train = train.drop(['SibSp','Parch'], axis=1)
test    = test.drop(['SibSp','Parch'], axis=1)

# plot
fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

# sns.factorplot('Family',data=train,kind='count',ax=axis1)
sns.countplot(x='Family', data=train, order=[1,0], ax=axis1)

# average of survived for those who had/didn't have any family member
family_perc = train[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)

axis1.set_xticklabels(["With Family","Alone"], rotation=0)

# Sex

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
train['Person'] = train[['Age','Sex']].apply(get_person,axis=1)
test['Person']    = test[['Age','Sex']].apply(get_person,axis=1)

# No need to use Sex column since we created Person column
train.drop(['Sex'],axis=1,inplace=True)
test.drop(['Sex'],axis=1,inplace=True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(train['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

train = train.join(person_dummies_titanic)
test    = test.join(person_dummies_test)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

# sns.factorplot('Person',data=train,kind='count',ax=axis1)
sns.countplot(x='Person', data=train, ax=axis1)

# average of survived for each Person(male, female, or child)
person_perc = train[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])

train.drop(['Person'],axis=1,inplace=True)
test.drop(['Person'],axis=1,inplace=True)

# Pclass

# sns.factorplot('Pclass',data=train,kind='count',order=[1,2,3])
sns.factorplot('Pclass','Survived',order=[1,2,3], data=train,size=5)

# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(train['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

train.drop(['Pclass'],axis=1,inplace=True)
test.drop(['Pclass'],axis=1,inplace=True)

train = train.join(pclass_dummies_titanic)
test    = test.join(pclass_dummies_test)