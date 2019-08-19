# Title: C4T3_caret_script

# Last update: 6.24.19

# File: C4T3_caret_script.R
# Project name: Multiple models for sentiment analysis


###############
# Project Notes
###############

# Summarize project: 

# Summarize top model and/or filtered dataset
# The top model was rfFit5 used with RFE dataset.



###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())

# get working directory
getwd()
? getwd  # get help
# set working directory
setwd("C:/Users/Megan/Documents/Data_Analytics/C4T3/")
dir()


################
# Load packages
################

install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("dplyr")
install.packages("mlbench")
install.packages("Hmisc")
install.packages("randomForest")
install.packages("doParallel")
install.packages("tidyverse")
devtools::install_github("ropensci/plotly")

library(caret)
library(corrplot)
library(readr)
library(dplyr)
library(mlbench)
library(Hmisc)
library(randomForest)
library(doParallel)
library(tidyverse)
library(ggplot2)
library(plotly)

#####################
# Parallel processing
#####################

#--- for Win ---#
detectCores()  # detect number of cores (2)
cl <- makeCluster(2)  # select number of cores
registerDoParallel(cl) # register cluster
getDoParWorkers()  # confirm number of cores being used by RStudio
# Stop Cluster. After performing your tasks, make sure to stop your cluster.

stopCluster(cl)
registerDoSEQ()



###############
# Import data
##############

#--- Load raw datasets ---#

## Load Train/Existing data (Dataset 1)
iphonematrix <-
  read.csv(
    "iphone_smallmatrix_labeled_8d.csv",
    stringsAsFactors = FALSE,
    header = T
  )
class(iphonematrix)  # "data.frame"


## Load Predict/New data (Dataset 2) ---#
iphonelargematrix <-
  read.csv(
    "iphoneLargeMatrix.csv",
    stringsAsFactors = FALSE,
    header = T
  )


#--- Load preprocessed datasets that have been saved ---#
#read back in files



################
# Evaluate data
################

#--- Dataset 1 ---#

str(iphonematrix)  #12973 obs. of  59 variables
#all columns are integers. will need to convert some or all later to factors

head(iphonematrix)
names(iphonematrix)
summary(iphonematrix)
#iphonesentiment
#Min.   :0.000  
#1st Qu.:3.000  
#Median :5.000  
#Mean   :3.725  
#3rd Qu.:5.000  
#Max.   :5.000 

#0: very negative 
#1: negative 
#2: somewhat negative
#3: somewhat positive
#4: positive
#5: very positive

# plot
plot_ly(iphonematrix, x= ~iphonematrix$iphonesentiment, name = 'iPhone Sentiment', type='histogram') %>%
  layout(title = "iPhone Sentiment",
         xaxis = list(title = "Sentiment Rating"),
         yaxis = list (title = "Count"))
#highest frequency is of rating of 5 at 7000+
#second highest is 0 at 2000 but huge gap between top two - class imbalance towards 5
plot_ly(iphonematrix, x= ~iphonematrix$iphone, type='histogram')
plot_ly(iphonematrix, x= ~iphonematrix$iphoneperpos, type='histogram')
plot_ly(iphonematrix, x= ~iphonematrix$iphoneperneg, type='histogram')
plot_ly(iphonematrix, x= ~iphonematrix$iphoneperunc, type='histogram')

# check for missing values
anyNA(iphonematrix)
#[1] FALSE
is.na(iphonematrix)
# remove or exclude missing values
na.omit(DatasetName$ColumnName) # Drops any rows with missing values and omits them forever.
na.exclude(DatasetName$ColumnName) # Drops any rows with missing values, but keeps track of where they were.

#--- Dataset 2 ---#
str(iphonelargematrix) 

anyNA(iphonelargematrix)
is.na(iphonelargematrix)
#only na is iphonesentiment, as expected

#############
# Preprocess
#############

#--- Dataset 1 ---#
#create corr matrix
options(max.print=10000)
coriphone <- cor(iphonematrix)
corrplot(coriphone, order = "hclust")
coriphone


# handle missing values (if applicable)
#na.omit(ds$ColumnName)
#na.exclude(ds$ColumnName)
#ds$ColumnName[is.na(ds$ColumnName)] <- mean(ds$ColumnName,na.rm = TRUE)

? na.omit  # returns object if with incomplete cases removed
? na.exclude

#after feature selection, update data types
iphonematrix$iphonesentiment <- as.ordered(iphonematrix$iphonesentiment)
str(iphonematrix)
#iphonesentiment: Ord.factor w/ 6 levels "1"<"2"<"3"<"4"<..: 1 1 1 1 1 5 5 1 1 1 ...

iphoneCOR$iphonesentiment <- as.ordered(iphoneCOR$iphonesentiment)
iphoneNZV$iphonesentiment <- as.ordered(iphoneNZV$iphonesentiment)
iphoneRFE$iphonesentiment <- as.ordered(iphoneRFE$iphonesentiment)

#--- Dataset 2 ---#
#remove ID
iphonelargematrix$id <- NULL



###############
# Save datasets
###############

# after ALL preprocessing, save a new version of the dataset

write.csv(iphoneCOR, file = "iphoneCOR.csv")
write.csv(iphoneNZV, file = "iphoneNZV.csv")
write.csv(iphoneRFE, file = "iphoneRFE.csv")

################
# Sampling
################


# ---- Sampling ---- #




##########################
# Feature Selection (FS) & Removal
##########################

# Three primary methods
# 1. Filtering
# 2. Wrapper methods (e.g., RFE caret)
# 3. Embedded methods (e.g., varImp)

###########
# Filtering
###########


#remove features based on correlation
#remove highly correlation ( > 0.9) values
hc <- findCorrelation(coriphone, cutoff = abs(0.9))
hc <- sort(hc)
iphoneCOR <- iphonematrix[,-c(hc)]
#new df with highly correlated values removed
str(iphoneCOR)
#'data.frame':	12973 obs. of  46 variables:
#'
#remove near-zero var features
nzvMetrics <- nearZeroVar(iphonematrix, saveMetrics = TRUE)
nzvMetrics
#most all features not referring to iphone have nzv = TRUE

#return matrix of nzv values
nzv <- nearZeroVar(iphonematrix, saveMetrics = FALSE) 
nzv

iphoneNZV <- iphonematrix[,-nzv]
str(iphoneNZV)
#data.frame':	12973 obs. of  12 variables:


############
# caret RFE
############

# lmFuncs - linear model
# rfFuncs - random forests
# nbFuncs - naive Bayes
# treebagFuncs - bagged trees


## ---- rf ---- ##

#sample the data (1000 rows) before using RFE
set.seed(123)
iphoneSample <- iphonematrix[sample(1:nrow(iphonematrix), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58],
                  iphoneSample$iphonesentiment,
                  sizes=(1:58),
                  rfeControl=ctrl)

# Get results
rfeResults
#The top 5 variables (out of 19):
#  iphone, googleandroid, iphonedispos, iphonedisneg, samsunggalaxy

# Plot results
plot(rfeResults, type=c("g", "o"))
varImp(rfeResults)
#optimal # of features = 19

# create new data set with rfe recommended features
iphoneRFE <- iphonematrix[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- iphonematrix$iphonesentiment

str(iphoneRFE)
#'data.frame':	12973 obs. of  20 variables:


##############################
# Variable Importance (varImp)
##############################

# varImp is evaluated in the model train/fit section


# ---- Conclusion ---- #

#

##########################
# Feature Engineering
##########################

#---Dataset 1---

# create a new dataset that will be used for recoding sentiment
#1: negative
#2: somewhat negative
#3: somewhat positive
#4: positive
iphoneRC <- iphonematrix
# recode sentiment to combine factor levels 1 & 2 and 5 & 6 
iphoneRC$iphonesentiment <- recode(iphoneRC$iphonesentiment, '1' = 1, '2' = 1, '3' = 2, '4' = 3, '5' = 4, '6' = 4)
# inspect results
summary(iphoneRC)
str(iphoneRC)
# make iphonesentiment a factor [not ordered]
iphoneRC$iphonesentiment <- as.factor(iphoneRC$iphonesentiment)

# create a new dataset that will be used for recoding sentiment incorporating RFE data
#1: negative
#2: somewhat negative
#3: somewhat positive
#4: positive
iphoneRC_RFE <- iphoneRFE
# recode sentiment to combine factor levels 1 & 2 and 5 & 6 
iphoneRC_RFE$iphonesentiment <- recode(iphoneRC_RFE$iphonesentiment, '1' = 1, '2' = 1, '3' = 2, '4' = 3, '5' = 4, '6' = 4)
# inspect results
summary(iphoneRC_RFE)
str(iphoneRC_RFE)
# make iphonesentiment a factor [not ordered]
iphoneRC_RFE$iphonesentiment <- as.factor(iphoneRC_RFE$iphonesentiment)

#---Dataset 2----
#apply same RFE to dataset based on top performing model (rfFitRC2)
iphonelargematrixRFE <- iphonelargematrix[,predictors(rfeResults)]

# add the dependent variable back in
iphonelargematrixRFE$iphonesentiment <- iphonelargematrix$iphonesentiment

str(iphonelargematrixRFE)
#'data.frame':	30032 obs. of  20 variables

##################
# Train/test sets
##################

# set random seed
set.seed(998)
# create the training partition that is 70% of total obs
inTraining_all <-
  createDataPartition(iphonematrix$iphonesentiment, p = 0.70, list = FALSE)
# create training/testing dataset
trainSet_all <- iphonematrix[inTraining_all, ]
testSet_all <- iphonematrix[-inTraining_all, ]
# verify number of obs
nrow(trainSet_all) # 9083
nrow(testSet_all)  # 3890
str(trainSet_all$iphonesentiment)

#create train/test sets with no zero var data set
# set random seed
set.seed(998)
# create the training partition that is 70% of total obs
inTraining_nozv <-
  createDataPartition(iphoneNZV$iphonesentiment, p = 0.70, list = FALSE)
# create training/testing dataset
trainSet_nozv <- iphoneNZV[inTraining_nozv, ]
testSet_nozv <- iphoneNZV[-inTraining_nozv, ]
# verify number of obs
nrow(trainSet_nozv) # 9083
nrow(testSet_nozv)  # 3890

#create train/test sets with filtered correlation data set
# set random seed
set.seed(123)
# create the training partition that is 70% of total obs
inTraining_COR <-
  createDataPartition(iphoneCOR$iphonesentiment, p = 0.70, list = FALSE)
# create training/testing dataset
trainSet_COR <- iphoneCOR[inTraining_COR, ]
testSet_COR <- iphoneCOR[-inTraining_COR, ]
# verify number of obs
nrow(trainSet_COR) # 9083
nrow(testSet_COR)  # 3890



#create train/test sets for RFE data set
# set random seed
set.seed(123)
# create the training partition that is 70% of total obs
inTraining_RFE <-
  createDataPartition(iphoneRFE$iphonesentiment, p = 0.70, list = FALSE)
# create training/testing dataset
trainSet_RFE <- iphoneRFE[inTraining_RFE, ]
testSet_RFE <- iphoneRFE[-inTraining_RFE, ]
# verify number of obs
nrow(trainSet_RFE) # 9083
nrow(testSet_RFE)  # 3890


#create train/test sets for new feature engineered set with 4 factors
# set random seed
set.seed(123)
# create the training partition that is 70% of total obs
inTraining_RC <-
  createDataPartition(iphoneRC$iphonesentiment, p = 0.70, list = FALSE)
# create training/testing dataset
trainSet_RC <- iphoneRC[inTraining_RC, ]
testSet_RC <- iphoneRC[-inTraining_RC, ]
# verify number of obs
nrow(trainSet_RC) # 9083
nrow(testSet_RC)  # 3890

#create train/test sets for new feature engineered set with 4 factors and RFE set
# set random seed
set.seed(123)
# create the training partition that is 70% of total obs
inTraining_RC_RFE <-
  createDataPartition(iphoneRC_RFE$iphonesentiment, p = 0.70, list = FALSE)
# create training/testing dataset
trainSet_RC_RFE <- iphoneRC_RFE[inTraining_RC_RFE, ]
testSet_RC_RFE <- iphoneRC_RFE[-inTraining_RC_RFE, ]
# verify number of obs
nrow(trainSet_RC) # 9083
nrow(testSet_RC)  # 3890

##################
# PCA
##################

# create object containing centered, scaled PCA components from training set
# exclude the dependent variable and set threshold to .95
preprocessParams <- preProcess(trainSet_all[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)
#PCA needed 25 components to capture 95 percent of the variance

#check for threshold at 0.8
preprocessParams2 <- preProcess(trainSet_all[,-59], method=c("center", "scale", "pca"), thresh = 0.8)
print(preprocessParams2)
#PCA needed 11 components to capture 80 percent of the variance

# use predict to apply pca parameters, create training, exclude dependent
train.pca <- predict(preprocessParams, trainSet_all[,-59])

# add the dependent to training
train.pca$iphonesentiment <- trainSet_all$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependent
test.pca <- predict(preprocessParams, testSet_all[,-59])

# add the dependent to testing
test.pca$iphonesentiment <- testSet_all$iphonesentiment

# inspect results
str(train.pca)
str(test.pca)

################
# Train control
################

# set 10 fold cross validation
fitControl <-
  trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 1,
    allowParallel = TRUE
  )



##############
# Train model
##############


## ------- KKNN ------- ##
set.seed(123)

#using weighted KNN (KKNN)
KKNN1 <- train(
  iphonesentiment ~ . ,
  data = trainSet_all,
  method = 'kknn',
  preProcess = c('center', 'scale'),
  trControl = fitControl,
  tuneLength = 3
)

KKNN1
#kmax  Accuracy   Kappa    
#5     0.3056202  0.1472761
#7     0.3088156  0.1514216
#9     0.3126699  0.1549957

KKNN2 <- train(
  iphonesentiment ~ . ,
  data = trainSet_all,
  method = 'kknn',
  preProcess = c('center', 'scale'),
  trControl = fitControl,
  tuneLength = 10
)

#kmax  Accuracy   Kappa    
# 5    0.3035378  0.1466532
# 7    0.3062897  0.1510029
# 9    0.3105797  0.1546382
#11    0.3169692  0.1599068
#13    0.3180685  0.1601664
#15    0.3229128  0.1633621
#17    0.3267654  0.1666531
#19    0.3287486  0.1674684
#21    0.3331507  0.1686797
#23    0.3345817  0.1702030

## ------- C5.0 ------- ##

set.seed(123)

C50Fit1 <-
  train(
    iphonesentiment  ~ .,
    data = trainSet_all,
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#   model  winnow  trials  Accuracy   Kappa    
#  rules  FALSE    1      0.7686906  0.5496494
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 1, model = rules and winnow = FALSE.

#with pre-processing
C50Fit2 <-
  train(
    iphonesentiment  ~ .,
    data = trainSet_all,
    preProcess = c('center', 'scale'),
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7627447  0.5375200

#****correlated features removed data set****
C50Fit3 <-
  train(
    iphonesentiment  ~ .,
    data = trainSet_COR,
    preProcess = c('center', 'scale'),
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7707801  0.5546851

#near zero var features removed
C50Fit4 <-
  train(
    iphonesentiment  ~ .,
    data = trainSet_nozv,
    preProcess = c('center', 'scale'),
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7519447  0.5136300

#RFE feature selection dataset
C50Fit5 <-
  train(
    iphonesentiment  ~ .,
    data = trainSet_RFE,
    preProcess = c('center', 'scale'),
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7679159  0.5477357

C50FitRC <-
  train(
    iphonesentiment  ~ .,
    data = trainSet_RC,
    preProcess = c('center', 'scale'),
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#model  winnow  trials  Accuracy   Kappa 
#tree   FALSE    1      0.8456531  0.6146241

C50Fitpca <-
  train(
    iphonesentiment  ~ .,
    data = train.pca,
    preProcess = c('center', 'scale'),
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#model  winnow  trials  Accuracy   Kappa 
#tree   FALSE    1      0.7575748  0.5296080

## ------- SVM ------- ##

set.seed(123)

SVMFit1 <-
  train(
    iphonesentiment  ~ .,
    data = trainSet_all,
    method = 'svmLinear2',
    trControl = fitControl
)

#cost  Accuracy   Kappa    
#0.25  0.6995510  0.3915980
#0.50  0.6997696  0.3924467
#1.00  0.7003202  0.3936253

#with pre-processing
SVMFit2 <-
  train(
    iphonesentiment  ~ .,
    data = trainSet_all,
    preProcess = c('center', 'scale'),
    method = 'svmLinear2',
    trControl = fitControl
  )

#cost  Accuracy   Kappa    
#0.25  0.6998825  0.3913208
#0.50  0.7007643  0.3930591
#1.00  0.7010929  0.3940456

SVMFit3 <-
  train(
    iphonesentiment  ~ .,
    data = trainSet_all,
    preProcess = c('center', 'scale'),
    method = 'svmLinear2',
    trControl = fitControl,
    tuneLength = 5
  )

#cost  Accuracy   Kappa    
#0.25  0.6996628  0.3906539
#0.50  0.6999925  0.3914216
#1.00  0.7020853  0.3958090
#2.00  0.7036268  0.3993607
#4.00  0.7032964  0.3987402

## ------- rf ------- ##
set.seed(123)

rfFit1 <-
  train(
    iphonesentiment ~ .,
    data = trainSet_all,
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 2
  )

#mtry  Accuracy   Kappa    
#2    0.6996586  0.3695889
#58    0.7604302  0.5436388


#with pre-processing and higher tuneLength
rfFit2 <-
  train(
    iphonesentiment ~ .,
    data = trainSet_all,
    preProcess = c('center', 'scale'),
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.6998784  0.3700211
#20    0.7735371  0.5611619
#39    0.7656114  0.5506382
#58    0.7608759  0.5443177

#COR dataset
rfFit3 <-
  train(
    iphonesentiment ~ .,
    data = trainSet_COR,
    preProcess = c('center', 'scale'),
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.6905164  0.3432595
#16    0.7730897  0.5609781
#30    0.7668135  0.5524426
#45    0.7624090  0.5462714

#near zero var dataset
rfFit4 <-
  train(
    iphonesentiment ~ .,
    data = trainSet_nozv,
    preProcess = c('center', 'scale'),
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.7576793  0.5216241
#5    0.7552564  0.5225370
#8    0.7504130  0.5161933
#11    0.7452386  0.5086009

#RFE dataset
rfFit5 <-
  train(
    iphonesentiment ~ .,
    data = trainSet_RFE,
    preProcess = c('center', 'scale'),
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.7347773  0.4628578
#7    0.7754029  0.5661489
#13    0.7697893  0.5590734
#19    0.7642844  0.5511691

#model using new recoded dv
rfFitRC <-
  train(
    iphonesentiment ~ .,
    data = trainSet_RC,
    preProcess = c('center', 'scale'),
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.7777176  0.3767692
#20    0.8507113  0.6273873
#39    0.8466370  0.6209557
#58    0.8431134  0.6152351

#model using new recoded dv with RFE
rfFitRC2 <-
  train(
    iphonesentiment ~ .,
    data = trainSet_RC_RFE,
    preProcess = c('center', 'scale'),
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.8087605  0.4901994
#7    0.8501573  0.6280909
#13    0.8474057  0.6241873
#19    0.8439917  0.6182971

varImp(rfFitRC2)
#rf variable importance

#variables are sorted by maximum importance across the classes
#                 1       2      3       4
#samsunggalaxy 21.121 12.9915 80.486 100.000
#iphone        76.984 27.0066 48.586  86.759
#googleandroid 11.697 25.9099 84.050  80.568
#iphonedispos  13.424  0.0000 56.281  63.387
#iphonedisunc   9.506  2.3656 44.582  61.330
#iphoneperpos  15.426  3.7223 48.303  25.865
#iphonedisneg   7.907  6.0004 31.513  26.491
#iphoneperunc   4.337  4.0485 27.144  20.623
#iphoneperneg   9.960  0.3386 25.593  21.243
#iphonecampos  11.935  2.3393 23.901  21.243
#iphonecamunc   8.914  4.1836 23.183  15.666
#htcphone      20.219 13.7768 23.122  23.164
#iphonecamneg  10.451  7.9669 19.875  16.045
#htccampos     15.286  8.4564 13.566  15.469
#htcdisunc      7.120  7.3260 13.684   9.198
#sonyxperia    12.265  8.9895  7.907  12.957
#htcperpos     10.309  6.5413  9.908  12.747
#htccamneg      9.943  7.9483 12.674   9.960
#ios           11.194  8.9698  9.343  12.249

rfFitPCA <-
  train(
    iphonesentiment ~ .,
    data = train.pca,
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.7593291  0.5362688
#9    0.7596575  0.5374491
#17    0.7594376  0.5370040
#25    0.7589978  0.5362824

##---------- nb ------------##
set.seed(123)

nbFit1 <-
  train(
    iphonesentiment ~ .,
    data = trainSet_all,
    preProcess = c('center', 'scale'),
    method = "naive_bayes",
    trControl = fitControl,
    tuneLength = 5
  )
#usekernel  Accuracy    Kappa       
#FALSE      0.07508049   0.039055897
#TRUE      0.10042262  -0.007520374

#not good model type for this data set, likely due to number of numeric features

##---------- adaboost ------------##

adaFit1 <-
  train(
    iphonesentiment ~ .,
    data = trainSet_all,
    preProcess = c('center', 'scale'),
    method = "AdaBag",
    trControl = fitControl,
    tuneLength = 5
  )
#maxdepth  mfinal  Accuracy   Kappa   
#5         200     0.7330165  0.4574917

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were mfinal = 200 and maxdepth = 5.

##--- Compare metrics ---##

ModelFitResults <- resamples(list(C50 = C50Fit1, SVM = SVMFit2, rf = rfFit2, ada = adaFit1))
# output summary metrics for tuned models
summary(ModelFitResults)
#Models: C50, SVM, rf, ada 
#Number of resamples: 10 

#Accuracy 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C50 0.7494505 0.7636216 0.7702476 0.7691305 0.7766758 0.7821782    0
#SVM 0.6842684 0.6963939 0.7008273 0.7010929 0.7105527 0.7136564    0
#rf  0.7612761 0.7676211 0.7724024 0.7735371 0.7779925 0.7898790    0
#ada 0.7213656 0.7282489 0.7326733 0.7330165 0.7397408 0.7422907    0

#Kappa 
#        Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C50 0.5040504 0.5379596 0.5534092 0.5509575 0.5675270 0.5803128    0
#SVM 0.3519465 0.3817183 0.3969627 0.3940456 0.4143533 0.4243182    0
#rf  0.5351238 0.5491370 0.5577779 0.5611619 0.5699437 0.5971782    0
#ada 0.4273835 0.4467101 0.4580850 0.4574917 0.4727193 0.4799300    0

#top two models: C50 and RF

#compare best C5.0 models using every data set
ModelFitResults2 <- resamples(list(C50_all = C50Fit1, C50_COR = C50Fit3, C50_nozv = C50Fit4, C50_RFE = C50Fit5 ))
summary(ModelFitResults2)
#Accuracy 
#               Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C50_all  0.7494505 0.7636216 0.7702476 0.7691305 0.7766758 0.7821782    0
#C50_COR  0.7557756 0.7670705 0.7717272 0.7707801 0.7740161 0.7896476    0
#C50_nozv 0.7100331 0.7513767 0.7556411 0.7519447 0.7596366 0.7681319    0
#C50_RFE  0.7433921 0.7595710 0.7688479 0.7679159 0.7785757 0.7909791    0

#Kappa 
#             Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C50_all  0.5040504 0.5379596 0.5534092 0.5509575 0.5675270 0.5803128    0
#C50_COR  0.5171964 0.5448896 0.5585892 0.5546851 0.5629713 0.5971101    0
#C50_nozv 0.4260847 0.5100537 0.5218925 0.5134027 0.5284440 0.5487480    0
#C50_RFE  0.4884989 0.5273985 0.5528400 0.5477357 0.5697479 0.6019837    0

#C50_COR is best model for C5.0

#compare best C5.0 model and RF models
ModelFitResults3 <- resamples(list(C50_COR = C50Fit3, RF_all = rfFit2, RF_COR = rfFit3, RF_nozv = rfFit4, RF_RFE = rfFit5 ))
summary(ModelFitResults3)
#Accuracy 
#             Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C50_COR 0.7557756 0.7670705 0.7717272 0.7707801 0.7740161 0.7896476    0
#RF_all  0.7612761 0.7676211 0.7724024 0.7735371 0.7779925 0.7898790    0
#RF_COR  0.7643172 0.7670045 0.7723255 0.7730897 0.7773824 0.7887789    0
#RF_nozv 0.7491749 0.7527533 0.7588103 0.7576793 0.7611582 0.7676211    0
#RF_RFE  0.7544053 0.7700155 0.7787555 0.7754029 0.7810872 0.7887789    0

#Kappa 
#             Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C50_COR 0.5171964 0.5448896 0.5585892 0.5546851 0.5629713 0.5971101    0
#RF_all  0.5351238 0.5491370 0.5577779 0.5611619 0.5699437 0.5971782    0
#RF_COR  0.5396083 0.5466343 0.5599955 0.5609781 0.5698304 0.5968753    0
#RF_nozv 0.5017562 0.5105659 0.5244207 0.5216241 0.5273909 0.5410505    0
#RF_RFE  0.5189455 0.5519835 0.5746500 0.5661489 0.5829021 0.5963663    0


#--- Save/load top performing model ---#

saveRDS(C50Fit3, "C50Fit_COR.rds")
saveRDS(rfFit5, "rfFit_RFE.rds")
# load and name model



############################
# Predict testSet/validation
############################


C50Pred <- predict(C50Fit3, testSet_COR)
postResample(C50Pred, testSet_COR$iphonesentiment) 
#Accuracy     Kappa 
#0.7583548 0.5267358 

C50ConfusionMatrix <- confusionMatrix(C50Pred, testSet_COR$iphonesentiment)
C50ConfusionMatrix
#Confusion Matrix and Statistics

#         Reference
#Prediction    1    2    3    4    5    6
#         1  394    0    0    3    4    7
#         2    0    0    0    0    0    0
#         3    0    0   22    0    0    0
#         4    2    0    1  175    3    1
#         5    4    0    3    1  135   30
#         6  188  117  110  177  289 2224

#Overall Statistics

#Accuracy : 0.7584          
#95% CI : (0.7446, 0.7717)
#No Information Rate : 0.5815          
#P-Value [Acc > NIR] : < 2.2e-16       

#Kappa : 0.5267          

#Mcnemar's Test P-Value : NA              

#Statistics by Class:

#                     Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
#Sensitivity            0.6701  0.00000 0.161765  0.49157  0.31323   0.9832
#Specificity            0.9958  1.00000 1.000000  0.99802  0.98901   0.4588
#Pos Pred Value         0.9657      NaN 1.000000  0.96154  0.78035   0.7163
#Neg Pred Value         0.9443  0.96992 0.970527  0.95119  0.92037   0.9516
#Prevalence             0.1512  0.03008 0.034961  0.09152  0.11080   0.5815
#Detection Rate         0.1013  0.00000 0.005656  0.04499  0.03470   0.5717
#Detection Prevalence   0.1049  0.00000 0.005656  0.04679  0.04447   0.7982
#Balanced Accuracy      0.8329  0.50000 0.580882  0.74480  0.65112   0.7210


rfPred <- predict(rfFit5, testSet_RFE)
postResample(rfPred, testSet_RFE$iphonesentiment)
#Accuracy     Kappa 
#0.7748072 0.5629674  

RFConfusionMatrix <- confusionMatrix(rfPred, testSet_RFE$iphonesentiment)
RFConfusionMatrix

#Confusion Matrix and Statistics

#           Reference
#Prediction    1    2    3    4    5    6
#         1  378    0    1    0    5   10
#         2    1    0    0    0    0    0
#         3    0    1   17    0    0    2
#         4    2    0    0  238    3    4
#         5    4    0    1    2  143    8
#         6  203  116  117  116  280 2238

#Overall Statistics

#Accuracy : 0.7748           
#95% CI : (0.7613, 0.7879)
#No Information Rate : 0.5815          
#P-Value [Acc > NIR] : < 2.2e-16       

#Kappa :0.563          

#Mcnemar's Test P-Value : NA              

#Statistics by Class:

#                     Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
#Sensitivity           0.64286 0.0000000 0.125000  0.66854  0.33179   0.9894
#Specificity           0.99515 0.9997350 0.999201  0.99745  0.99566   0.4889
#Pos Pred Value        0.95939 0.0000000 0.850000  0.96356  0.90506   0.7290
#Neg Pred Value        0.93993 0.9699151 0.969251  0.96761  0.92283   0.9707
#Prevalence            0.15116 0.0300771 0.034961  0.09152  0.11080   0.5815
#Detection Rate        0.09717 0.0000000 0.004370  0.06118  0.03676   0.5753
#Detection Prevalence  0.10129 0.0002571 0.005141  0.06350  0.04062   0.7892
#Balanced Accuracy     0.81901 0.4998675 0.562100  0.83300  0.66373   0.7392

##Conclusion: RF model has best performance metrics using RFE data set. Both top models struggled with correctly predicting
#classes 2,3, and 5. Next will try feature engineering to improve sensitivity

#predict test set with reduced dv model/data
rfPred2 <- predict(rfFitRC2, testSet_RC)
postResample(rfPred2, testSet_RC$iphonesentiment)
#Accuracy     Kappa 
#0.8632391 0.6636410 

RFConfusionMatrix2 <- confusionMatrix(rfPred2, testSet_RC$iphonesentiment)
RFConfusionMatrix2
#Confusion Matrix and Statistics

#             Reference
#Prediction    1    2    3    4
#           1  417    0    0   11
#           2    0   23    0    0
#           3    1    0  243    7
#           4  287  113  113 2675

#Overall Statistics

#Accuracy : 0.8632         
#95% CI : (0.852, 0.8739)
#No Information Rate : 0.6923         
#P-Value [Acc > NIR] : < 2.2e-16      

#Kappa : 0.6636         

#Mcnemar's Test P-Value : NA             

#Statistics by Class:

#                     Class: 1 Class: 2 Class: 3 Class: 4
#Sensitivity            0.5915 0.169118  0.68258   0.9933
#Specificity            0.9965 1.000000  0.99774   0.5714
#Pos Pred Value         0.9743 1.000000  0.96813   0.8391
#Neg Pred Value         0.9168 0.970778  0.96895   0.9744
#Prevalence             0.1812 0.034961  0.09152   0.6923
#Detection Rate         0.1072 0.005913  0.06247   0.6877
#Detection Prevalence   0.1100 0.005913  0.06452   0.8195
#Balanced Accuracy      0.7940 0.584559  0.84016   0.7824

#conclusion: slightly better accuracy and kappa, but sensitivity still low for class 2 but overall improved

C50PredRC <- predict(C50FitRC, testSet_RC)
postResample(C50PredRC, testSet_RC$iphonesentiment) 
#Accuracy     Kappa 
#0.8439589 0.6107665 

rfPredRC <- predict(rfFitRC, testSet_RC)
postResample(rfPredRC, testSet_RC$iphonesentiment) 
#Accuracy     Kappa 
#0.8485861 0.6236138 

###############################
# Predict new data (Dataset 2)
###############################

#predict with both recoded and non-recoded RF model [need to keep scale consistent for galaxy sentiment]
rfPredfinal <- predict(rfFitRC2, iphonelargematrixRFE)
rfPredfinal2 <- predict(rfFit5, iphonelargematrixRFE)

summary(rfPredfinal)
#1     2     3     4 
#12927  1096  1583 14426

#delete blank iphonesentiment column
iphonelargematrix$iphonesentiment <- NULL

#add predictions to dataset
iphonelargematrix <- mutate(iphonelargematrix,rfPredfinal)
iphonelargematrix <- mutate(iphonelargematrix,rfPredfinal2)

#recode back to original scale
iphonelargematrix$rfPredfinal2 <- recode(iphonelargematrix$rfPredfinal2, '1' = 0, '2' = 1, '3' = 2, '4' = 3, '5' = 4, '6' = 5)

colnames(iphonelargematrix)
#rename columns
iphonelargematrix <- iphonelargematrix  %>% rename(iphonesentiment = rfPredfinal, iphonesentiment2 = rfPredfinal2)

#save to file
write.csv(iphonelargematrix,file = "iphoneLargeMatrix.csv")



