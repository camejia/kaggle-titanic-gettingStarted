# TODO: Insert a standard file docstring here

import pandas as pd
import numpy as np
import pylab
# import csv as csv
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the training file into a dataframe
train_df = pd.read_csv('C:\Users\Chris\Documents\camejia_svn\Python\kaggle\\titanic-gettingStarted\\train.csv', header=0)

# Load the testing file into a dataframe
test_df = pd.read_csv('C:\Users\Chris\Documents\camejia_svn\Python\kaggle\\titanic-gettingStarted\\test.csv', header=0)

all_df = pd.concat([train_df, test_df])

# Column headings are PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# For the columns I plan to use, make sure the data is all valid
print 'Pclass.unique(): ', all_df.Pclass.unique()
print 'Sex.unique(): ', all_df.Sex.unique()
print 'SibSp.unique(): ', all_df.SibSp.unique()
print 'Parch.unique(): ', all_df.Parch.unique()
print 'Age.isnull().any()', all_df.Age.isnull().any()
print 'Fare.isnull().any()', all_df.Fare.isnull().any()

# I know survival rates were very different for females vs. male,
# so I'll process them separately

for Sex in ['female', 'male']:
    for Pclass in [1, 2, 3]:
        print 'Sex: ', Sex, ', Pclass: ', Pclass
        print train_df[(train_df.Sex == Sex) & (train_df.Pclass == Pclass)].corr()

# Find cases where a variable is correlated to Survived with |x| > 0.3
# Sex == 'female' & Pclass == 3 -> Fare
# Sex == 'male' & Pclass == 2 -> Age and Parch

f3d = train_df[(test_df.Sex == 'female') & (test_df.Pclass == 3) & (test_df.Survived == 0)]
f3s = train_df[(test_df.Sex == 'female') & (test_df.Pclass == 3) & (test_df.Survived == 1)]

m2d = train_df[(test_df.Sex == 'male') & (test_df.Pclass == 2) & (test_df.Survived == 0)]
m2s = train_df[(test_df.Sex == 'male') & (test_df.Pclass == 2) & (test_df.Survived == 1)]

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


all_female1_df = all_df[(all_df.Sex == 'female') & (all_df.Pclass == 1)]
all_female2_df = all_df[(all_df.Sex == 'female') & (all_df.Pclass == 2)]
all_female3_df = all_df[(all_df.Sex == 'female') & (all_df.Pclass == 3)]

all_male1_df = all_df[(all_df.Sex == 'male') & (all_df.Pclass == 1)]
all_male2_df = all_df[(all_df.Sex == 'male') & (all_df.Pclass == 2)]
all_male3_df = all_df[(all_df.Sex == 'male') & (all_df.Pclass == 3)]

# I'll just use the median Age and Fare for a given Pclass
# Note that I use train and test data (i.e. not just train data)
# to calculate the value to fill in

all_female1_df = all_female1_df.fillna(all_female1_df.median())
all_female2_df = all_female2_df.fillna(all_female2_df.median())
all_female3_df = all_female3_df.fillna(all_female3_df.median())

all_male1_df = all_male1_df.fillna(all_male1_df.median())
all_male2_df = all_male2_df.fillna(all_male2_df.median())
all_male3_df = all_male3_df.fillna(all_male3_df.median())

# Look at the correlations for each Sex and Pclass combination
# Which have large values (|x|>0.25) in the Survived column (or row)?

print 'all_female1_df.corr()', all_female1_df.corr()
all_female2_df.corr()
all_female3_df.corr()
all_male1_df.corr()
all_male2_df.corr()
all_male3_df.corr()

pylab.figure()
pylab.plot(all_male2_df.Age, all_male2_df.Parch, 'ko')
pylab.plot(all_male2_df.Age, all_male2_df.Parch, 'rx')
pylab.title('Females')
pylab.xlabel('Age')
pylab.ylabel('Parch')
pylab.grid()
pylab.show()

test_df.Survived = test_df.PassengerId
test_df[test_df.Sex == 'female'].Survived = 1
test_df[test_df.Sex == 'male'].Survived = 0

test_df[(test_df.Sex == 'female') & (test_df.Pclass == 3) & ((test_df.Fare > 20.7) | (test_df.Age > 38.5))] = 0
test_df[(test_df.Sex == 'male') & (test_df.Pclass == 2) & (test_df.Age < 12)] = 1

# sklearn.linear_model.LogisticRegression

# Data cleanup

## Convert all strings to integer classifiers.
## female = 0, male = 1
#train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
#test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
#
#temp1 = train_df.Embarked.dropna()
#temp2 = test_df.Embarked.dropna()


### Data cleanup
### TRAIN DATA
##
### I need to fill in the missing values of the data and make it complete.
##
##
### Embarked from 'C', 'Q', 'S'
### Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.
##
### All missing Embarked -> just make them embark from most common place
##if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
##    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values
##
##Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
##Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
##train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int
##
### All the ages with no data -> make the median of all Ages
##median_age = train_df['Age'].dropna().median()
##if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
##    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age
##
### Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
##train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
##
##
### TEST DATA
##
### I need to do the same with the test data now, so that the columns are the same as the training data
### I need to convert all strings to integer classifiers:
### female = 0, Male = 1
##test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
##
### Embarked from 'C', 'Q', 'S'
### All missing Embarked -> just make them embark from most common place
##if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
##    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
### Again convert all Embarked strings to int
##test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)
##
##
### All the ages with no data -> make the median of all Ages
##median_age = test_df['Age'].dropna().median()
##if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
##    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age
##
### All the missing Fares -> assume median of their respective class
##if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
##    median_fare = np.zeros(3)
##    for f in range(0,3):                                              # loop 0 to 2
##        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
##    for f in range(0,3):                                              # loop 0 to 2
##        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]
##
### Collect the test data's PassengerIds before dropping it
##ids = test_df['PassengerId'].values
### Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
##test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
##
##
### The data is now ready to go. So lets fit to the train, then predict to the test!
### Convert back to a numpy array
##train_data = train_df.values
##test_data = test_df.values
##
##
##print 'Training...'
##forest = RandomForestClassifier(n_estimators=100)
##forest = forest.fit( train_data[0::,1::], train_data[0::,0] )
##
##print 'Predicting...'
##output = forest.predict(test_data).astype(int)
##
##
##predictions_file = open("myfirstforest.csv", "wb")
##open_file_object = csv.writer(predictions_file)
##open_file_object.writerow(["PassengerId","Survived"])
##open_file_object.writerows(zip(ids, output))
##predictions_file.close()
##print 'Done.'
