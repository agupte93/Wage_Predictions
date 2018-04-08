######################## DISCOVERY PHASE##############################

######################### read data into R################
getwd()
#setwd("C:/Users/KP/Desktop/Fall 2017/MIS 620 Big Data/Final Project")

#library(data.table)
#library(dplyr)
#library(ggplot2)
library(caret)
library(mice)


alldata <- read.csv(file = "2016Data_withSelectedColumns.csv")
str(alldata)

View(alldata)  # View the data set

#check the column names of the data frame
colnames(alldata)


# #check for outliers in the 'odds' column for player 1 using boxplot
# 
# boxplot_Player1_odd<- boxplot(alldata_to_process$Player_1_odd, xlab = "", ylab= "player1 odds", main = "Outliers for Odds")
# 
# 
# #check for outliers
# 
# outlier_Player1_odd <- data.frame() #initialising storage for outliers
# 
# if(length(boxplot_Player1_odd$out) > 0){
#   for(n in 1:length(boxplot_Player1_odd$out)){
#     pt <-data.frame(value=boxplot_Player1_odd$out[n],group=0) 
#     outlier_Player1_odd<-rbind(outlier_Player1_odd,pt) 
#   }
# }
# 
# #check for outliers in the 'odds' column for player 2 using boxplot
# 
# boxplot_Player2_odd<- boxplot(alldata_to_process$Player_2_odd, xlab = "Player2 odds", ylab= "player2 odd")
# 
# 
# #check for outliers
# 
# outlier_Player2_odd <- data.frame() #initialising storage for outliers
# 
# if(length(boxplot_Player2_odd$out) > 0){
#   for(n in 1:length(boxplot_Player2_odd$out)){
#     pt <-data.frame(value=boxplot_Player2_odd$out[n],group=0) 
#     outlier_Player2_odd<-rbind(outlier_Player2_odd,pt) 
#   }
# }

################# DATA PREPARATION##################

# #remove all Player 1 odd outlier rows from data frame
# 
# alldata_to_process_without_player1odd_outlier = alldata_to_process[!(alldata_to_process$Player_1_odd  %in% outlier_Player1_odd$value),]
# 
# #remove all NAs from player 1 odd and take only valid values for mean calculation
# 
# player1_odd_for_mean = na.omit(alldata_to_process_without_player1odd_outlier$Player_1_odd)
# 
# #calculate mean for player 1 odd
# 
# mean_player1_odd=mean(player1_odd_for_mean)
# mean_player1_odd=round(mean_player1_odd,2)
# 
# #find outlier range for player 1 odd
# minimum_player1_odd_outlier = min(outlier_Player1_odd$value)
# 
# #replace all outlier player 1 odd values with the mean value for player 1 odd
# for(i in 1:nrow(alldata_to_process))
# {
#   if (alldata_to_process$Player_1_odd[i] > minimum_player1_odd_outlier && !is.na(alldata_to_process$Player_1_odd[i])) {
#     alldata_to_process$Player_1_odd[i] = mean_player1_odd
#   }
# }
# 
# 
# #remove all Player 2 odd outlier rows from data frame
# 
# alldata_to_process_without_player2odd_outlier = alldata_to_process[!(alldata_to_process$Player_2_odd  %in% outlier_Player2_odd$value),]
# 
# #remove all NAs from player 2 odd and take only valid values for mean calculation
# 
# player2_odd_for_mean = na.omit(alldata_to_process_without_player2odd_outlier$Player_2_odd)
# 
# #calculate mean for player 2 odd
# 
# mean_player2_odd=mean(player2_odd_for_mean)
# mean_player2_odd=round(mean_player2_odd,2)
# 
# #find outlier range for player 2 odd
# outlier_Player2_odd=subset(outlier_Player2_odd, outlier_Player2_odd$value > 0)
# minimum_player2_odd_outlier = min(outlier_Player2_odd$value)
# 
# #replace all outlier player 2 odd values with the mean value for player 2 odd
# for(i in 1:nrow(alldata_to_process))
# {
#   if (alldata_to_process$Player_2_odd[i] > minimum_player2_odd_outlier && !is.na(alldata_to_process$Player_2_odd[i])) {
#     alldata_to_process$Player_2_odd[i] = mean_player2_odd
#   }
# }

#check data rows with DDRS=NA (people with age<5)

DDRSNA<- subset(alldata, is.na(alldata$DDRS))

#imputing NA values using Mice package
set.seed(2)
md.pattern(alldata)
write.csv(md.pattern(alldata), "test.csv")


# library(VIM)
# aggr_plot <- aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))

data_new=data.frame(alldata$CITWP, alldata$SERIALNO)
micedData <- mice(data_new,m=2,maxit=2,meth='pmm')
micedData<- complete(micedData,1)

View(micedData)
summary(sampledata)
summary(micedData)


write.csv(micedData, "MicedData.csv")

###################  MODEL PLANNING  ##########################


#let's split the data into training and tests sets
#create training 

library(caret)
library(pROC)
library(rpart)
library(rpart.plot)
library(ROCR)

#lets use the credit dataset predicting defaulting on a loan
alldata <- read.csv("MicedData.csv")

alldata = alldata[,-1]
alldata$player_1_score<-sample(1:50,nrow(alldata),rep=TRUE)
alldata$player_2_score<-sample(2:60,nrow(alldata),rep=TRUE)

alldata = alldata[1:30000,]
y <- alldata$Player_1_Wins #keep your DV away from all processing and filtering to avoid overfitting!
x <- alldata[,c(1,3,4,5,6,7,8,9,18,19,20,23,24)] #only first 16 columns (predictors)
set.seed(200)
#lets grab 70% of data for training and 30% for test sets
#will sample proportional to base rate of the DV (we have imbalanced data more no then yes)
inTrain<-createDataPartition(y=y, p=.70, list=FALSE)
y.train <- y[inTrain]
x.train <- x[inTrain,]
y.test<- y[-inTrain]
x.test <- x[-inTrain,]

#check composition
table(y.train)
table(y.test)

Train.data<-alldata[inTrain,]
Test.data<-alldata[-inTrain,]
#lets split out using index of training and test sets created above, uses row index

#twoClassSummary is built in function with ROC, Sensitivity and Specificity
ctrl <- trainControl(method="cv",classProbs=TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE)


################# MODEL BUILDING and RESULTS#################


## Decision Trees

modelLookup("rpart")

str(Train.data)


m.rpart <- train(Player_1_Wins ~   doubles  + player_1_score + player_2_score + retired_player + cancelled_game + walkover + Player_1_odd + Player_2_odd , 
                 trControl = ctrl,
                 metric = "Accuracy", #using AUC to find best performing parameters
                 preProc = c("range", "nzv"), #scale from 0 to 1 and from columns with zero variance
                 data = Train.data, 
                 method = "rpart")

m.rpart


p.rpart <- predict(m.rpart,Test.data)
p.rpart
confusionMatrix(p.rpart,Test.data$Player_1_Wins) 


pred <- prediction(as.numeric(p.rpart), Test.data$Player_1_Wins)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc")@y.values
confusionMatrix(p.rpart, Test.data$Player_1_Wins)

plot(perf, lwd=2, xlab="False Positive Rate (FPR)",
     ylab="True Positive Rate (TPR)")
abline(a=0, b=1, col="gray50", lty=3)


## corresponding AUC score
auc <- performance(pred, "auc")
auc <- unlist(slot(auc, "y.values"))
auc



# Make a simple decision tree 
fit <- rpart(Player_1_Wins ~ country + day + tournament_name + doubles + player_1_name + player_2_name + player_1_score + player_2_score + retired_player + cancelled_game + walkover + Player_1_odd + Player_2_odd, 
             method="class", 
             data=Train.data,
             control=rpart.control(minsplit=1),  ###minsplit=1 means we want at least one split in the tree
             parms=list(split='information'))

summary(fit)

#lets prune the tree to avoid overfitting using cross validation

printcp(fit) #display crossvalidated error for each tree size
plotcp(fit) #plot cv error

#select CP with lowest crossvalidated error 

#we can grab this from the plotcp table automatically with 
opt.cp <- fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]

#lets prune the tree
fit.pruned <- prune(fit, cp=opt.cp)


summary(fit.pruned)


#lets review the final tree
rpart.plot(fit.pruned)

p.rpart <- predict(fit.pruned,Test.data)

score <- p.rpart[, c("Yes")]
actual_class <- Test.data$Player_1_Wins == "Yes"
pred <- prediction(score, actual_class)  #creates a prediction object which is passed on to performance object
perf <- performance(pred, "tpr", "fpr")

plot(perf, lwd=2, xlab="False Positive Rate (FPR)",
     ylab="True Positive Rate (TPR)")
abline(a=0, b=1, col="gray50", lty=3)

## corresponding AUC score
auc <- performance(pred, "auc")
auc <- unlist(slot(auc, "y.values"))
auc



##################################NAIVE BAYES#######################3

library(e1071)

nb_model <- naiveBayes(Player_1_Wins ~ country + day + tournament_name + doubles + player_1_name + player_2_name + player_1_score + player_2_score + retired_player + cancelled_game + walkover + Player_1_odd + Player_2_odd, 
                       Train.data)

# display model
nb_model


# predict with testdata
results <- predict (nb_model,Test.data[,-10], type='raw')

# display results
results

score <- results[, c("Yes")]
actual_class <- Test.data$Player_1_Wins == "Yes"
pred <- prediction(score, actual_class)  #creates a prediction object which is passed on to performance object
perf <- performance(pred, "tpr", "fpr")

plot(perf, lwd=2, xlab="False Positive Rate (FPR)",
     ylab="True Positive Rate (TPR)")
abline(a=0, b=1, col="gray50", lty=3)

## corresponding AUC score
auc <- performance(pred, "auc")
auc <- unlist(slot(auc, "y.values"))
auc




#########################Laplace smoothing######################################################

nb_model1 = naiveBayes(Player_1_Wins ~ country + day + tournament_name + doubles + player_1_name + player_2_name + player_1_score + player_2_score + retired_player + cancelled_game + walkover + Player_1_odd + Player_2_odd, 
                       Train.data, laplace=1) 
nb_model1

results1 <- predict (nb_model1,Test.data,type = 'raw')
results1

score <- results1[, c("Yes")]
actual_class <- Test.data$Player_1_Wins == "Yes"
pred <- prediction(score, actual_class)  #creates a prediction object which is passed on to performance object
perf <- performance(pred, "tpr", "fpr")

plot(perf, lwd=2, xlab="False Positive Rate (FPR)",
     ylab="True Positive Rate (TPR)")
abline(a=0, b=1, col="gray50", lty=3)

## corresponding AUC score
auc <- performance(pred, "auc")
auc <- unlist(slot(auc, "y.values"))
auc





#############--------NEAURAL NETWORK----------------

modelLookup("nnet")

m.nnet <- train(Player_1_Wins ~   doubles + player_1_name   +    player_2_name  + player_1_score + player_2_score + retired_player + cancelled_game + walkover + awarded_player + Player_1_odd + Player_2_odd, 
                trControl = ctrl2,
                metric = "Accuracy", #using AUC to find best performing parameters
                preProc = c("range", "nzv"), #scale from 0 to 1 and from columns with zero variance
                data = Train.data, 
                method = "nnet")

m.nnet
plot(m.nnet)
p.nnet <- predict(m.nnet,dummy.test)
confusionMatrix(p.nnet,dummy.test$country_destination)

###############-----------------BAGGING ------------------
#some meta-learning examples
##BAGGING - bootstrapping is used to create many training sets and simple models are trained on each and combined
##many small decision trees
#install.packages("ipred")
library(ipred)

m.bag <- train(Player_1_Wins ~   doubles + player_1_name   +    player_2_name  + player_1_score + player_2_score + retired_player + cancelled_game + walkover + awarded_player + Player_1_odd + Player_2_odd, 
               trControl = ctrl,
               metric = "Accuracy", #using AUC to find best performing parameters
               preProc = c("range", "nzv"), #scale from 0 to 1 and from columns with zero variance
               data = Train.data, 
               method = "treebag")
m.bag
p.bag<- predict(m.bag,Test.data)
p.bag
confusionMatrix(p.bag,Test.data$Player_1_Wins)


pred <- prediction(as.numeric(p.bag), Test.data$Player_1_Wins)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc")@y.values
confusionMatrix(p.rf, Test.data$Player_1_Wins)

plot(perf, lwd=2, xlab="False Positive Rate (FPR)",
     ylab="True Positive Rate (TPR)")
abline(a=0, b=1, col="gray50", lty=3)


## corresponding AUC score
auc <- performance(pred, "auc")
auc <- unlist(slot(auc, "y.values"))
auc

###########----------------RANDOM FOREST-------------------------
#random forest approach to many classification models created and voted on
#less prone to ovrefitting and used on large datasets
library(randomForest)
set.seed(10)

RF<- randomForest(Player_1_Wins ~   doubles  + player_1_score + player_2_score + retired_player + cancelled_game + walkover + awarded_player + Player_1_odd + Player_2_odd, 
                  data=Train.data, importance=TRUE, proximity=FALSE, 
                  ntree=10000, keep.forest=TRUE)

p.rf<- predict(RF,Test.data)
p.rf

print(p.rf)

pred <- prediction(as.numeric(p.rf), Test.data$Player_1_Wins)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc")@y.values
confusionMatrix(p.rf, Test.data$Player_1_Wins)

plot(perf, lwd=2, xlab="False Positive Rate (FPR)",
     ylab="True Positive Rate (TPR)")
abline(a=0, b=1, col="gray50", lty=3)


## corresponding AUC score
auc <- performance(pred, "auc")
auc <- unlist(slot(auc, "y.values"))
auc

########### with CARET RANDOM FOREST##############
m.rf <- train(Player_1_Wins ~   doubles + player_1_name   +    player_2_name  + player_1_score + player_2_score + retired_player + cancelled_game + walkover + awarded_player + Player_1_odd + Player_2_odd, 
              trControl = ctrl,
              metric = "Accuracy", #using AUC to find best performing parameters
              preProc = c("range", "nzv"), #scale from 0 to 1 and from columns with zero variance
              data = Train.data, 
              method = c("rf") )
m.rf
p.rf<- predict(m.rf,Test.data)
confusionMatrix(p.rf,Test.data$Player_1_Wins)

pred <- prediction(as.numeric(p.rf), Test.data$Player_1_Wins)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc")@y.values
confusionMatrix(p.rf, Test.data$Player_1_Wins)

plot(perf, lwd=2, xlab="False Positive Rate (FPR)",
     ylab="True Positive Rate (TPR)")
abline(a=0, b=1, col="gray50", lty=3)


## corresponding AUC score
auc <- performance(pred, "auc")
auc <- unlist(slot(auc, "y.values"))
auc


#######################################------SVM-----##########################

library(kernlab)

set.seed(200)


svm_classifier <- ksvm(Player_1_Wins ~   doubles + player_1_name   +    player_2_name  + player_1_score + player_2_score + retired_player + cancelled_game + walkover + awarded_player + Player_1_odd + Player_2_odd,  
                       data = Train.data,
                       kernel = "vanilladot", scaled=TRUE)

svm_classifier

##  Evaluating model performance ----
# predictions on testing dataset
p_svm_classifier <- predict(svm_classifier, Test.data)
head(p_svm_classifier)

table(p_svm_classifier, Test.data$Player_1_Wins)

#use caret confusion matrix
confusionMatrix(p_svm_classifier,Test.data$Player_1_Wins) #72.44% AVG Accuracy

pred <- prediction(as.numeric(p_svm_classifier), Test.data$Player_1_Wins)
perf <- performance(pred,"tpr","fpr")
performance(pred,"auc")@y.values
confusionMatrix(p_svm_classifier, Test.data$Player_1_Wins)

plot(perf, lwd=2, xlab="False Positive Rate (FPR)",
     ylab="True Positive Rate (TPR)")
abline(a=0, b=1, col="gray50", lty=3)


## corresponding AUC score
auc <- performance(pred, "auc")
auc <- unlist(slot(auc, "y.values"))
auc


