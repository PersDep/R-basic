#! /usr/bin/Rscript


create_train_test <- function(data, size, train = TRUE) {
	n_row = nrow(data)
	total_row = size * n_row
	train_sample <- 1 : total_row
	if (train == TRUE) return (data[train_sample, ])
	return (data[-train_sample, ])
}

make_predictions_and_verify <- function(data, fit) {
	predict_survivors <- predict(fit, data, type = 'class')
	table_mat <- table(data$survived, predict_survivors)
	print(table_mat)
	accuracy <- sum(diag(table_mat)) / sum(table_mat)
	print(paste('Accuracy', accuracy))
}


set.seed(777)
path <- 'https://raw.githubusercontent.com/guru99-edu/R-Programming/master/titanic_data.csv'
titanic <-read.csv(path)

# Glimpse at the data
print('Head')
head(titanic)
print('Tail')
tail(titanic)

# Shuffle data for fair train/test distribution
shuffle_index <- sample(1:nrow(titanic))
titanic <- titanic[shuffle_index,]
print('Shuffled head')
head(titanic)

library(dplyr)
# Drop variables & convert to proper data types
titanic_linear <- mutate(select(titanic, -c(home.dest, cabin, name, x, ticket, embarked)),
			sex = factor(sex, levels = c('male', 'female'), labels = c(0, 1)))
titanic_linear <- subset(titanic_linear, age != '?')
titanic_linear <- subset(titanic_linear, fare != '?')
titanic_linear$age <- as.integer(titanic_linear$age)
titanic_linear$fare <- as.numeric(titanic_linear$fare)
titanic <- mutate(select(titanic, -c(home.dest, cabin, name, x, ticket)),
		pclass = factor(pclass, levels = c(1, 2, 3), labels = c('Upper', 'Middle', 'Lower')),
		survived = factor(survived, levels = c(0, 1), labels = c('No', 'Yes')))
titanic <- subset(titanic, age != '?')
titanic <- subset(titanic, fare != '?')
titanic$age <- as.integer(titanic$age)
titanic$fare <- as.numeric(titanic$fare)
print('Cleaned data')
glimpse(titanic)

data_train <- create_train_test(titanic, 0.8, train = TRUE)
data_test <- create_train_test(titanic, 0.8, train = FALSE)

print('Train & test dimensions')
dim(data_train)
dim(data_test)
print('Survived in train & test data')
prop.table(table(data_train$survived))
prop.table(table(data_test$survived))

library(rpart)
library(rpart.plot)
print('Default tree')
fit <- rpart(survived~., data = data_train, method = 'class')
rpart.plot(fit, extra = 106)
print('Predict on train data')
make_predictions_and_verify(data_train, fit)
print('Predict on test data')
make_predictions_and_verify(data_test, fit)
print('Metrics on default tree')
printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 

print('Tuned tree')
tfit <- rpart(survived~., data = data_train, method = 'class', maxdepth = 2)
rpart.plot(tfit, extra = 106)
print('Predict on train data')
make_predictions_and_verify(data_train, tfit)
print('Predict on test data')
make_predictions_and_verify(data_test, tfit)
print('Metrics on tuned tree')
printcp(tfit) # display the results
plotcp(tfit) # visualize cross-validation results

# Linear regression experiment
print('Cleaned linear data')
glimpse(titanic_linear)
data_train <- create_train_test(titanic_linear, 0.8, train = TRUE)
data_test <- create_train_test(titanic_linear, 0.8, train = FALSE)
linearModel <- lm(survived~., data = data_train)
predict_survived <- predict(linearModel, newdata = data_test)
for (i in 1:length(predict_survived)) if (predict_survived[i] < 0.5) predict_survived[i] <- 0 else predict_survived[i] <- 1
table_mat <- table(data_test$survived, predict_survived)
print(table_mat)
accuracy <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Linear accuracy', accuracy))
