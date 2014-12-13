# TODO: Insert a standard file docstring here

import pandas as pd
import numpy as np
from sklearn import svm

# Load the training file into a dataframe
train_df = pd.read_csv('C:\Users\Chris\Documents\GitHub\kaggle-titanic-gettingStarted\\train.csv', header=0)

# Load the testing file into a dataframe
test_df = pd.read_csv('C:\Users\Chris\Documents\GitHub\kaggle-titanic-gettingStarted\\test.csv', header=0)

all_df = pd.concat([train_df, test_df])

# Column headings are PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# For the columns I plan to use, make sure the data is all valid
print 'Pclass.unique(): ', all_df.Pclass.unique()
print 'Sex.unique(): ', all_df.Sex.unique()
print 'SibSp.unique(): ', all_df.SibSp.unique()
print 'Parch.unique(): ', all_df.Parch.unique()
print 'Age.isnull().any()', all_df.Age.isnull().any()
print 'Fare.isnull().any()', all_df.Fare.isnull().any()

# Should try to avoid using terms that are highly correlated
all_df.corr()
# As expected Fare and Pclass are correlated, maybe combine into one?

# I know survival rates were very different for females vs. male,
# so I'll process them separately to fill in missing data

train_df['AgeFilled'] = train_df.Age
train_df['FareFilled'] = train_df.Fare
test_df['AgeFilled'] = test_df.Age
test_df['FareFilled'] = test_df.Fare

for Sex in ['female', 'male']:
    for Pclass in [1, 2, 3]:
        train_df.loc[(train_df.Sex == Sex) & (train_df.Pclass == Pclass) & (train_df.AgeFilled.isnull()), 'AgeFilled'] = all_df.Age[(all_df.Sex == Sex) & (all_df.Pclass == Pclass)].median()
        train_df.loc[(train_df.Sex == Sex) & (train_df.Pclass == Pclass) & (train_df.FareFilled.isnull()), 'FareFilled'] = all_df.Fare[(all_df.Sex == Sex) & (all_df.Pclass == Pclass)].median()
        test_df.loc[(test_df.Sex == Sex) & (test_df.Pclass == Pclass) & (test_df.AgeFilled.isnull()), 'AgeFilled'] = all_df.Age[(all_df.Sex == Sex) & (all_df.Pclass == Pclass)].median()
        test_df.loc[(test_df.Sex == Sex) & (test_df.Pclass == Pclass) & (test_df.FareFilled.isnull()), 'FareFilled'] = all_df.Fare[(all_df.Sex == Sex) & (all_df.Pclass == Pclass)].median()

# Check that I got rid of all of the NaN's
print 'train_df.AgeFilled.isnull().any()', train_df.AgeFilled.isnull().any()
print 'train_df.FareFilled.isnull().any()', train_df.FareFilled.isnull().any()
print 'test_df.AgeFilled.isnull().any()', test_df.AgeFilled.isnull().any()
print 'test_df.FareFilled.isnull().any()', test_df.FareFilled.isnull().any()

# Convert string 'Sex' to a number 'Gender' so I can use scikit-learn
train_df['Gender'] = train_df.Sex.map( {'female': 0, 'male': 1} )
test_df['Gender'] = test_df.Sex.map( {'female': 0, 'male': 1} )

train_np = train_df[['Pclass', 'SibSp', 'Parch', 'AgeFilled', 'FareFilled', 'Gender']].values
test_np = test_df[['Pclass', 'SibSp', 'Parch', 'AgeFilled', 'FareFilled', 'Gender']].values
train_surv_np = train_df.Survived.values

# Check that no NaN's made it through
print 'np.isnan(train_np).any()', np.isnan(train_np).any()
print 'np.isnan(test_np).any()', np.isnan(test_np).any()
print 'np.isnan(train_surv_np).any()', np.isnan(train_surv_np).any()

print 'Training...'
clf = svm.SVC()
clf.fit(train_np, train_surv_np)

print 'Predicting...'
test_surv_np = clf.predict(test_np)

test_df['Survived'] = test_surv_np

test_df[['PassengerId', 'Survived']].to_csv('C:\Users\Chris\Documents\GitHub\kaggle-titanic-gettingStarted\scikit-learn.csv', index=False)