######################## WAGES PREDICTION DATA MODELS##############################

######################### LOAD LIBRARIES ################
getwd()
library(rpart.plot)
library(caret)
library(mice)
library(earth)
library(ggplot2)
library(gridExtra)
library(ggthemes)
library(dplyr)
library(corrplot)
library(mlbench)
library(parallel)
library(doParallel)
library(VIM)
library(purrr)
library(tidyr)
library(ggplot2)



## LOAD DATA in R environment ##

alldata <- read.csv(file = "2016Data_withSelectedColumns.csv")
str(alldata)


View(alldata)  # View the data set

#check the column names of the data frame
colnames(alldata)

#Create Unique Identifier

newID <-
  paste(alldata$SERIALNO, "_", alldata$SPORDER, "_", alldata$PUMA)

#remove 1st 3 fields from alldata

alldata = alldata[-c(1:3)]

# add newID column
alldata = cbind(newID, alldata)

#remove rows where WAGP is 0
alldata <- subset(alldata, alldata$WAGP != 0)

#remove rows where INDP is NA< represents people with age 16 or below
alldata <- subset(alldata, !is.na(alldata$INDP))

#remove rows where DDRS is NA< represents people with age 5 or below
alldata <- subset(alldata, !is.na(alldata$DDRS))


#remove rows where AGEP > 10
alldata <- subset(alldata, alldata$AGEP > 10)

saveRDS(alldata, "alldata.rds")

#separate out numerical and categorical variable

alldata_numerical <- alldata[, c(2:4, 6, 28:29, 39, 56:57, 59, 67:68, 72)]
alldata_numerical <- alldata_numerical[1:100000, ]
saveRDS(alldata_numerical, "alldata_numerical.rds")

alldata_categorical <- alldata[-c(2:4, 6, 28:29, 39, 56:57, 59, 67:68, 72)]

#removing categorical variables based on data understanding
alldata_categorical <-
  alldata_categorical[-c(
    5,
    6,
    7,
    8,
    10,
    11,
    14,
    15,
    16,
    17:23,
    24,
    25,
    27,
    31,
    32,
    35:43,
    46:48,
    49,
    53,
    57:65,
    68:72,
    74,
    75,
    79:84,
    87,
    89,
    91:93,
    95,
    96,
    97,
    98:112
  )]
str(alldata_categorical)

alldata_categorical <- alldata_categorical[1:100000, ]
saveRDS(alldata_categorical, "alldata_categorical.rds")


#find categorical columns which have NA and need to be imputed
x <- unlist(lapply(alldata_categorical, function(x)
  any(is.na(x))))

#find numerical columns which have NA and need to be imputed
y <- unlist(lapply(alldata_numerical, function(x)
  any(is.na(x))))
#str(alldata_categorical)

#imputing NA values using Mice package
set.seed(2)
md.pattern(alldata)

aggr_plot <-
  aggr(
    alldata,
    col = c('navyblue', 'red'),
    numbers = TRUE,
    sortVars = TRUE,
    labels = names(data),
    cex.axis = .7,
    gap = 3,
    ylab = c("Histogram of missing data", "Pattern")
  )

md.pattern(alldata_numerical)
md.pattern(alldata_categorical)

#this is how to use mice in parallel to save time------------ Parallelize MICE
cores <- 4 #don't use more than 4 cores
cl <-  makeForkCluster(cores)
clusterSetRNGStream(cl, 489) #set seed for everymember of cluster
registerDoParallel(cl)

#how many sets to impute
msets = 2

#using foreach to seperate mice runs and recombine results
#for large datasets this can speed up imputation dramatically!

micedData_numerical <-
  mice(alldata_numerical[1:100000, c(4, 6, 7, 13)],
       m = 2,
       maxit = 2,
       meth = 'pmm')
micedData_numerical <- complete(micedData_numerical, 1)
saveRDS(micedData_numerical, "micedData_numerical.rds")
View(micedData_numerical)
str(micedData_numerical)



micedData_categorical <-
  mice(alldata_categorical[1:100000, c(5, 6)],
       m = 2,
       maxit = 2,
       meth = 'rf')
micedData_categorical_5and6 <- complete(micedData_categorical, 1)
saveRDS(micedData_categorical_5and6,
        "micedData_categorical_5and6.rds")


micedData_categorical_7_10_11_13 <-
  mice(alldata_categorical[1:100000, c(7, 10, 11, 13)],
       m = 2,
       maxit = 2,
       meth = 'rf')
micedData_categorical_7_10_11_13 <-
  complete(micedData_categorical_7_10_11_13, 1)
saveRDS(micedData_categorical_7_10_11_13,
        "micedData_categorical_7_10_11_13.rds")

micedData_categorical_17_22 <-
  mice(alldata_categorical[1:100000, c(17, 22)],
       m = 2,
       maxit = 1,
       meth = 'rf')
micedData_categorical_17_22 <-
  complete(micedData_categorical_17_22, 1)
saveRDS(micedData_categorical_17_22,
        "micedData_categorical_17_22.rds")


micedData_categorical_25 <-
  mice(alldata_categorical[1:100000, c(8, 25, 30, 31)],
       m = 2,
       maxit = 2,
       meth = 'sample')
micedData_categorical_25 <- complete(micedData_categorical_25, 1)
saveRDS(micedData_categorical_25,
        "micedData_categorical_25_8_30_31.rds")

alldata_categorical$MIGSP[is.na(alldata_categorical$MIGSP)] <- 0

#completed categorical merged dataset with imputations
complete_miced_categorical <-
  cbind(
    micedData_categorical_5and6,
    micedData_categorical_7_10_11_13,
    micedData_categorical_25_8_30_31,
    micedData_categorical_17_22,
    alldata_categorical$MIGSP
  )
complete_categorical <-
  cbind(complete_miced_categorical, alldata_categorical[, c(1:4, 9, 12, 14, 15, 16, 18, 19, 20, 21, 23, 24, 27, 28, 29, 32)])
saveRDS(complete_categorical, "complete_categorical.rds")


complete_numerical <-
  cbind(micedData_numerical, alldata_numerical[, c(1, 2, 3, 5, 8, 9, 10, 11, 12)])
saveRDS(complete_numerical, "complete_numerical.rds")
#remove near zero variance


final_imputed_data <- cbind(complete_numerical, complete_categorical)
saveRDS(final_imputed_data, "final_imputed.rds")

NearZeroNumerical <-
  nearZeroVar(complete_numerical,
              allowParallel = TRUE,
              saveMetrics = TRUE)
str(NearZeroNumerical)
micedData_numerical_withoutzeroVar <-
  complete_numerical[-c(5, 8, 9, 10, 11)]
colnames(micedData_numerical_withoutzeroVar)

final_data <-
  cbind(micedData_numerical_withoutzeroVar, complete_categorical)
final_withoutNewID <- final_data[, -22]
lenstr(final_withoutNewID)
saveRDS(final_withoutNewID, "final_processed_data.rds")


#### Remove Highly Correlated Variables ####
# calculate correlation matrix
correlationMatrix <- cor(final_withoutNewID)
corrplot(correlationMatrix)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.75)
# print indexes of highly correlated attributes
print(highlyCorrelated)

#removing highly correlated variables
final_withoutNewID <- final_withoutNewID[, -c(highlyCorrelated)]
#Removing Income to Poverty Recode
final_withoutNewID <- final_withoutNewID[, -c(17)]

#Variable Importance Code Using Earth Package
m.earth <-
  earth(WAGP ~ ., data = final_withoutNewID) # finding important variables
ev <- evimp(m.earth)
plot(ev,
     cex.legend = 1,
     x.legend = nrow(x),
     y.legend = x[1, "nsubsets"])
plot(ev)
impvar_data <- final_withoutNewID[, c(27, 7, 5, 31, 32, 28, 33, 21, 20, 34, 6)]
colnames(impvar_data)

#Subset Wages in Range of $20,000 to $200,000

impvar_data <- subset(impvar_data, impvar_data$WAGP > 20000)
impvar_data <- subset(impvar_data, impvar_data$WAGP < 200000)


impvar_data %>%
  keep(is.numeric) %>%
  gather() %>%
  ggplot(aes(value)) +
  facet_wrap( ~ key, scales = "free") +
  geom_histogram()


# Plotting Boxplots to View Outliers in Important Variables

boxplot_column1 <-
  qplot(
    x = impvar_data$WAGP,
    y = impvar_data[, 1],
    geom = "boxplot" ,
    xlab = "",
    ylab = "SCHL"
  )
boxplot_column2 <-
  qplot(
    x = impvar_data$WAGP,
    y = impvar_data[, 2],
    geom = "boxplot" ,
    xlab = "",
    ylab = "WKHP"
  )
boxplot_column3 <-
  qplot(
    x = impvar_data$WAGP,
    y = impvar_data[, 3],
    geom = "boxplot" ,
    xlab = "",
    ylab = "AGEP"
  )
boxplot_column4 <-
  qplot(
    x = impvar_data$WAGP,
    y = impvar_data[, 4],
    geom = "boxplot" ,
    xlab = "",
    ylab = "ESR"
  )
boxplot_column5 <-
  qplot(
    x = impvar_data$WAGP,
    y = impvar_data[, 5],
    geom = "boxplot" ,
    xlab = "",
    ylab = "INDP"
  )
boxplot_column6 <-
  qplot(
    x = impvar_data$WAGP,
    y = impvar_data[, 6],
    geom = "boxplot" ,
    xlab = "",
    ylab = "SEX"
  )
boxplot_column7 <-
  qplot(
    x = impvar_data$WAGP,
    y = impvar_data[, 7],
    geom = "boxplot" ,
    xlab = "",
    ylab = "MSP"
  )
boxplot_column8 <-
  qplot(
    x = impvar_data$WAGP,
    y = impvar_data[, 8],
    geom = "boxplot" ,
    xlab = "",
    ylab = "CIT"
  )
boxplot_column9 <-
  qplot(
    x = impvar_data$WAGP,
    y = impvar_data[, 9],
    geom = "boxplot" ,
    xlab = "",
    ylab = "RAC1P"
  )

grid.arrange(
  boxplot_column1,
  boxplot_column2,
  boxplot_column3,
  boxplot_column4,
  boxplot_column5,
  boxplot_column6,
  boxplot_column7,
  boxplot_column8,
  boxplot_column9,
  ncol = 9
)





#Train Test Split
y <- impvar_data$WAGP
x <-
  impvar_data[, c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)] #only first 16 columns (predictors)

inTrain <- createDataPartition(y = y, p = .70, list = FALSE)
Train.data <- impvar_data[inTrain, ]
Test.data <- impvar_data[-inTrain, ]
na.omit(Train.data)
na.omit(Test.data)
summary(Train.data)
summary(Test.data)

#CV
ctrl <- trainControl(method = "cv", number = 5)


#Regular Linear Regression
lm.train <- train(
  WAGP ~ .,
  data = Train.data,
  method = "lm",
  tuneLength = 4,
  preProcess = c("scale", "center"),
  trControl = ctrl
)
summary(lm.train)
lm.predict <- predict(lm.train, Test.data)
RMSE(lm.predict, Test.data$WAGP)

# Decision Trees

dc.train <- train(
  WAGP ~ .,
  data = Train.data,
  method = "rpart",
  tuneLength = 4,
  preProcess = c("scale", "center"),
  trControl = ctrl
)
dc.train
dc.rpart <- rpart(WAGP ~ ., data = Train.data)

#very readable defaults
rpart.plot(dc.rpart)
opt.cp <-
  dc.rpart$cptable[which.min(dc.rpart$cptable[, "xerror"]), "CP"]

#lets prune the tree
dc.rpart.pruned <- prune(dc.rpart, cp = opt.cp)
rpart.plot(dc.rpart.pruned)
yhat.dc.rpart <- predict(dc.rpart.pruned, Test.data)
RMSE(yhat.dc.rpart, Test.data$WAGP)

#smooth spline regression
#gamSpline in caret will expand each predictor with smooth spline searching for df value
gam.train <- train(
  WAGP ~ .,
  data = Train.data,
  method = "gamSpline",
  tuneLength = 4,
  preProcess = c("scale", "center"),
  trControl = ctrl
)
gam.train

gam.predict <- predict(gam.train, Test.data)

RMSE(gam.predict, Test.data$WAGP)

#Lasso
lasso.train <- train(
  WAGP ~ .,
  data = Train.data,
  method = "lasso",
  tuneLength = 4,
  preProcess = c("scale", "center"),
  trControl = ctrl
)
lasso.train
lasso.predict <- predict(lasso.train, Test.data)

RMSE(lasso.predict, Test.data$WAGP)

#RF
rf.train <- train(
  WAGP ~ .,
  data = Train.data,
  method = "rf",
  tuneLength = 4,
  preProcess = c("scale", "center"),
  trControl = ctrl
)
plot(rf.train)
rf.predict <- predict(rf.train, Test.data)
RMSE(rf.predict, Test.data$WAGP)


#bagging tree
bag.train <-
  train(
    WAGP ~ .,
    data = Train.data,
    preProcess = c("scale", "center"),
    method = "treebag",
    tuneLength = 4,
    trControl = ctrl
  )


bag.train
plot(bag.train)
bag.predict <- predict(bag.train, Test.data)

RMSE(bag.predict, Test.data$WAGP)


#boosting
boost.train <- train(
  WAGP ~ .,
  data = Train.data,
  method = "gbm",
  tuneLength = 4,
  trControl = ctrl
)
boost.train
plot(boost.train)

boost.predict <- predict(boost.train, Test.data)

RMSE(boost.predict, Test.data$WAGP)




models_train <- list(
  "lm" = lm.train,
  "gam" = gam.train,
  "lasso" = lasso.train,
  "DT" = dc.train,
  "BaggingTree" = bag.train,
  "RF" = rf.train,
  "BoostingTree" = boost.train
)


data.resamples1 <- resamples(models_train)

summary(data.resamples1)

### VISUALIZING TRAIN PERFORMANCE ###
bwplot(data.resamples1, metric = "RMSE")
bwplot(data.resamples1, metric = "Rsquared")

### Resamples for Test Error(RMSE) and Rsquared ###
postResample(pred = lm.predict, obs = Test.data$WAGP)
postResample(pred = yhat.dc.rpart, obs = Test.data$WAGP)
postResample(pred = gam.predict, obs = Test.data$WAGP)
postResample(pred = bag.predict, obs = Test.data$WAGP)
postResample(pred = lasso.predict, obs = Test.data$WAGP)
postResample(pred = bag.predict, obs = Test.data$WAGP)
postResample(pred = boost.predict, obs = Test.data$WAGP)
postResample(pred = rf.predict, obs = Test.data$WAGP)
