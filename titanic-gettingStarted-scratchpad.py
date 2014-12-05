# TODO: Insert a standard file docstring here

import pandas as pd
import numpy as np
# import csv as csv
# from sklearn.ensemble import RandomForestClassifier

# Load the training file into a dataframe
train_df = pd.read_csv('C:\Users\Chris\Documents\camejia_svn\Python\kaggle\\titanic-gettingStarted\\train.csv', header=0)

# Load the testing file into a dataframe
test_df = pd.read_csv('C:\Users\Chris\Documents\camejia_svn\Python\kaggle\\titanic-gettingStarted\\test.csv', header=0)

all_df = pd.concat([train_df, test_df])

# First task is to fill in age

all_finite_age_df = all_df[np.isfinite(all_df['Age'])]

# Calculate median (or mean?) age as a function of passenger class (and sex?)
a1 = median(all_finite_age_df.Age[all_finite_age_df.Pclass == 1])
a2 = median(all_finite_age_df.Age[all_finite_age_df.Pclass == 2])
a3 = median(all_finite_age_df.Age[all_finite_age_df.Pclass == 3])

af1 = median(all_finite_age_df.Age[np.logical_and(all_finite_age_df.Pclass == 1, all_finite_age_df.Sex == 'female')])
af2 = median(all_finite_age_df.Age[np.logical_and(all_finite_age_df.Pclass == 2, all_finite_age_df.Sex == 'female')])
af3 = median(all_finite_age_df.Age[np.logical_and(all_finite_age_df.Pclass == 3, all_finite_age_df.Sex == 'female')])

am1 = median(all_finite_age_df.Age[np.logical_and(all_finite_age_df.Pclass == 1, all_finite_age_df.Sex == 'male')])
am2 = median(all_finite_age_df.Age[np.logical_and(all_finite_age_df.Pclass == 2, all_finite_age_df.Sex == 'male')])
am3 = median(all_finite_age_df.Age[np.logical_and(all_finite_age_df.Pclass == 3, all_finite_age_df.Sex == 'male')])

# Fill in the missing ages
train_df.loc[np.logical_and(train_df.Age.isnull(), train_df.Pclass == 1), 'Age'] = a1
train_df.loc[np.logical_and(train_df.Age.isnull(), train_df.Pclass == 2), 'Age'] = a2
train_df.loc[np.logical_and(train_df.Age.isnull(), train_df.Pclass == 3), 'Age'] = a3

train_female_df = train_df[train_df['Sex'] == 'female']
train_male_df = train_df[train_df['Sex'] == 'male']

train_female_survived_df = train_female_df[train_female_df["Survived"] == 1]
train_female_died_df = train_female_df[train_female_df["Survived"] == 0]
train_male_survived_df = train_male_df[train_male_df["Survived"] == 1]
train_male_died_df = train_male_df[train_male_df["Survived"] == 0]


figure()
plot(train_female_survived_df.Age, train_female_survived_df.Fare, 'ko')
plot(train_female_died_df.Age, train_female_died_df.Fare, 'rx')
title('Females')
xlabel('Age')
ylabel('Fare')
# legend('Survived', 'Died') # TBR
grid()

figure()
plot(train_male_survived_df.Age, train_male_survived_df.Fare, 'ko')
plot(train_male_died_df.Age, train_male_died_df.Fare, 'rx')
title('Males')
xlabel('Age')
ylabel('Fare')
# legend('Survived', 'Died') # TBR
grid()

# Data cleanup

# Convert all strings to integer classifiers.
# female = 0, male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

temp1 = train_df.Embarked.dropna()
temp2 = test_df.Embarked.dropna()


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
