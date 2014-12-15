# TODO: Insert a standard file docstring here

import pandas as pd
import numpy as np
from sklearn import svm
import pylab

# Load the training file into a dataframe
train_df = pd.read_csv('C:\Users\Chris\Documents\GitHub\kaggle-titanic-gettingStarted\\train.csv', header=0)

# Load the testing file into a dataframe
test_df = pd.read_csv('C:\Users\Chris\Documents\GitHub\kaggle-titanic-gettingStarted\\test.csv', header=0)

all_df = pd.concat([train_df, test_df])

# Just for fun, add a few plots

f3d = train_df[(train_df.Sex == 'female') & (train_df.Pclass == 3) & (train_df.Survived == 0)]
f3s = train_df[(train_df.Sex == 'female') & (train_df.Pclass == 3) & (train_df.Survived == 1)]

m2d = train_df[(train_df.Sex == 'male') & (train_df.Pclass == 2) & (train_df.Survived == 0)]
m2s = train_df[(train_df.Sex == 'male') & (train_df.Pclass == 2) & (train_df.Survived == 1)]

pylab.figure()
pylab.plot(f3s.Age, f3s.Fare, 'ko')
pylab.plot(f3d.Age, f3d.Fare, 'rx')
pylab.title('Females in 3rd Class')
pylab.xlabel('Age')
pylab.ylabel('Fare')
pylab.grid()
pylab.show()

pylab.figure()
pylab.plot(m2s.Age, m2s.Parch, 'ko')
pylab.plot(m2d.Age, m2d.Parch, 'rx')
pylab.title('Males in 2nd Class')
pylab.xlabel('Age')
pylab.ylabel('Parch')
pylab.grid()
pylab.show()

# Column headings are PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# For the columns I plan to use, make sure the data is all valid
print 'Pclass.unique(): ', all_df.Pclass.unique()
print 'Sex.unique(): ', all_df.Sex.unique()
print 'SibSp.unique(): ', all_df.SibSp.unique()
print 'Parch.unique(): ', all_df.Parch.unique()
print 'Age.isnull().any()', all_df.Age.isnull().any()
print 'Fare.isnull().any()', all_df.Fare.isnull().any()
print

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

# Convert string 'Sex' to a number 'Gender' so I can use scikit-learn
train_df['Gender'] = train_df.Sex.map( {'female': 0, 'male': 1} )
test_df['Gender'] = test_df.Sex.map( {'female': 0, 'male': 1} )

test_df['Survived'] = test_df.Gender
for Sex in ['female', 'male']:
    for Pclass in [1, 2, 3]:
        train_np = train_df[(train_df.Sex == Sex) & (train_df.Pclass == Pclass)][['FareFilled']].values
        test_np = test_df[(test_df.Sex == Sex) & (test_df.Pclass == Pclass)][['FareFilled']].values
        train_surv_np = train_df[(train_df.Sex == Sex) & (train_df.Pclass == Pclass)].Survived.values

        print 'Training...'
        clf = svm.SVC()
        clf.fit(train_np, train_surv_np)

        print 'Predicting...'
        test_surv_np = clf.predict(test_np)

        test_df.loc[(test_df.Sex == Sex) & (test_df.Pclass == Pclass), 'Survived'] = test_surv_np

test_df[['PassengerId', 'Survived']].to_csv('C:\Users\Chris\Documents\GitHub\kaggle-titanic-gettingStarted\scikit-learn.csv', index=False)
