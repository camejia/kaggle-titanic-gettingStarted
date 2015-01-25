# kaggle's Titanic: Machine Learning from Disaster, "Getting Started" Competition
# Based on http://trevorstephens.com/post/72916401642/titanic-getting-started-with-r

library(rattle)
library(rpart)
library(rpart.plot)
library(RColorBrewer)

# Set working directory and import datafiles
setwd("~/GitHub/kaggle-titanic-gettingStarted")
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Add 'everyone dies' prediction to the test set dataframe
test$Survived <- rep(0, 418)

# Write 'everyone dies' prediction to a csv file
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "theyallperish.csv", row.names = FALSE)

# Gender Model: all females survived, no males survived
test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1

# Write Gender Model prediction to a csv file
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "gendermodel.csv", row.names = FALSE)

# Create a new variable, Child, to indicate whether the passenger is below the age of 18:
train$Child <- 0
train$Child[train$Age < 18] <- 1

# Print proportions by Child and Sex
aggregate(Survived ~ Child + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

# Bin fares into groups, place result in Fare2
train$Fare2 <- '30+'
train$Fare2[train$Fare < 30 & train$Fare >= 20] <- '20-30'
train$Fare2[train$Fare < 20 & train$Fare >= 10] <- '10-20'
train$Fare2[train$Fare < 10] <- '<10'

# Print proportions by Fare2, Pclass, and Sex
aggregate(Survived ~ Fare2 + Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

# Most of the class 3 women who paid more than $20 for their ticket don't survive
test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1
test$Survived[test$Sex == 'female' & test$Pclass == 3 & test$Fare >= 20] <- 0

# Write Gender and Class Model prediction to a csv file
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "genderclassmodel.csv", row.names = FALSE)

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data=train, method="class")

# plot(fit)
# text(fit)

fancyRpartPlot(fit)
