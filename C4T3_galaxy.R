# Title: C4T3_caret_script

# Last update: 6.24.19

# File: C4T3_galaxy.R
# Project name: Multiple models for sentiment analysis


###############
# Project Notes
###############

# Summarize project: 

# Summarize top model and/or filtered dataset
# The top model was model_name used with ds_name.



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
galaxymatrix <-
  read.csv(
    "galaxy_smallmatrix_labeled_9d.csv",
    stringsAsFactors = FALSE,
    header = T
  )
class(galaxymatrix)  # "data.frame"

colnames(galaxymatrix)

## Load Predict/New data (Dataset 2) ---#
galaxylargematrix <-
  read.csv(
    "galaxyLargeMatrix.csv",
    stringsAsFactors = FALSE,
    header = T
  )
colnames(galaxylargematrix)
galaxylargematrix$id <- NULL

#--- Load preprocessed datasets that have been saved ---#
#read back in files
galaxyNZV <-
  read.csv(
    "galaxyNZV.csv"
  )
colnames(galaxyNZV)
galaxyNZV$X <- NULL

galaxyCOR <-
  read.csv(
    "galaxyCOR.csv"
  )
colnames(galaxyCOR)
galaxyCOR$X <- NULL

galaxyRFE <-
  read.csv(
    "galaxyRFE.csv"
  )
colnames(galaxyRFE)
galaxyRFE$X <- NULL

################
# Evaluate data
################

#--- Dataset 1 ---#

str(galaxymatrix)  #12911 obs. of  59 variables

head(galaxymatrix)
names(galaxymatrix)
summary(galaxymatrix)
#galaxysentiment
#Min.   :0.000  
#1st Qu.:3.000  
#Median :5.000  
#Mean   :3.829  
#3rd Qu.:5.000  
#Max.   :5.000 

#0: very negative 
#1: negative 
#2: somewhat negative
#3: somewhat positive
#4: positive
#5: very positive

# plot
plot_ly(galaxymatrix, x= ~galaxymatrix$galaxysentiment, name = 'Galaxy Sentiment', type='histogram') %>%
  layout(title = "Galaxy Sentiment",
         xaxis = list(title = "Sentiment Rating"),
         yaxis = list (title = "Count"))
#highest frequency is of rating of 5 at almost 8000
#second highest is 0 at close to 2000 but huge gap between top two - class imbalance towards 5
plot_ly(galaxymatrix, x= ~galaxymatrix$galaxy, type='histogram')
plot_ly(galaxymatrix, x= ~galaxymatrix$galaxyperpos, type='histogram')
plot_ly(galaxymatrix, x= ~galaxymatrix$galaxyperneg, type='histogram')
plot_ly(galaxymatrix, x= ~galaxymatrix$galaxyperunc, type='histogram')

# check for missing values
anyNA(galaxymatrix)
#[1] FALSE
is.na(galaxymatrix)
# remove or exclude missing values
na.omit(DatasetName$ColumnName) # Drops any rows with missing values and omits them forever.
na.exclude(DatasetName$ColumnName) # Drops any rows with missing values, but keeps track of where they were.

#--- Dataset 2 ---#


#############
# Preprocess
#############

#--- Dataset 1 ---#
#create corr matrix
options(max.print=10000)
corgalaxy <- cor(galaxymatrix)
corrplot(corgalaxy, order = "hclust")
corgalaxy


# handle missing values (if applicable)
#na.omit(ds$ColumnName)
#na.exclude(ds$ColumnName)
#ds$ColumnName[is.na(ds$ColumnName)] <- mean(ds$ColumnName,na.rm = TRUE)

? na.omit  # returns object if with incomplete cases removed
? na.exclude

#after feature selection, update data types
galaxymatrix$galaxysentiment <- as.factor(galaxymatrix$galaxysentiment)
str(galaxymatrix)
#galaxysentiment: Factor w/ 6 levels "0","1","2","3",..: 6 4 4 1 2 1 4 6 6 6 ...

galaxyCOR$galaxysentiment <- as.factor(galaxyCOR$galaxysentiment)
galaxyNZV$galaxysentiment <- as.factor(galaxyNZV$galaxysentiment)
galaxyRFE$galaxysentiment <- as.factor(galaxyRFE$galaxysentiment)

#--- Dataset 2 ---#
str(galaxylargematrix)


###############
# Save datasets
###############

# after ALL preprocessing, save a new version of the dataset

write.csv(galaxyCOR, file = "galaxyCOR.csv")
write.csv(galaxyNZV, file = "galaxyNZV.csv")
write.csv(galaxyRFE, file = "galaxyRFE.csv")

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
#--- Dataset 1 ---#

#remove features based on correlation
#remove highly correlation ( > 0.9) values
hc <- findCorrelation(corgalaxy, cutoff = abs(0.9))
hc <- sort(hc)
galaxyCOR <- galaxymatrix[,-c(hc)]
#new df with highly correlated values removed
str(galaxyCOR)
#'data.frame':	12911 obs. of  45 variables:
#'
#remove near-zero var features
nzvMetrics <- nearZeroVar(galaxymatrix, saveMetrics = TRUE)
nzvMetrics
#most all features not referring to galaxy have nzv = TRUE

#return matrix of nzv values
nzv <- nearZeroVar(galaxymatrix, saveMetrics = FALSE) 
nzv

galaxyNZV <- galaxymatrix[,-nzv]
str(galaxyNZV)
#data.frame':	12911 obs. of  12 variables:

#--- Dataset 2 ---#
galaxylargematrixNZV <- galaxylargematrix[,-nzv]

galaxylargematrixCOR <- galaxylargematrix[,-c(hc)]

############
# caret RFE
############

# lmFuncs - linear model
# rfFuncs - random forests
# nbFuncs - naive Bayes
# treebagFuncs - bagged trees


## ---- rf ---- ##
#--- Dataset 1 ---#
#sample the data (1000 rows) before using RFE
set.seed(123)
galaxySample <- galaxymatrix[sample(1:nrow(galaxymatrix), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 galaxysentiment) 
rfeResults <- rfe(galaxySample[,1:58],
                  galaxySample$galaxysentiment,
                  sizes=(1:58),
                  rfeControl=ctrl)

# Get results
rfeResults
#The top 5 variables (out of 18):
#  iphone, samsunggalaxy, googleandroid, htcphone, iphonedisunc

# Plot results
plot(rfeResults, type=c("g", "o"))
varImp(rfeResults)
#optimal # of features = 18

# create new data set with rfe recommended features
galaxyRFE <- galaxymatrix[,predictors(rfeResults)]

# add the dependent variable to galaxyRFE
galaxyRFE$galaxysentiment <- galaxymatrix$galaxysentiment

str(galaxyRFE)
#'data.frame':	12911 obs. of  19 variables:


#--- Dataset 2 ---#

galaxylargematrixRFE <- galaxylargematrix
galaxylargematrixRFE <- galaxylargematrix[,predictors(rfeResults)]

# add the dependent variable to galaxyRFE
galaxylargematrixRFE$galaxysentiment <- galaxylargematrix$galaxysentiment


##############################
# Variable Importance (varImp)
##############################

# varImp is evaluated in the model train/fit section


# ---- Conclusion ---- #

#

##########################
# Feature Engineering
##########################

#--- Dataset 1 ---#

# create a new dataset that will be used for recoding sentiment
#1: negative
#2: somewhat negative
#3: somewhat positive
#4: positive
galaxyRC <- galaxymatrix
# recode sentiment to combine factor levels 1 & 2 and 5 & 6 
galaxyRC$galaxysentiment <- recode(galaxyRC$galaxysentiment, '0'= 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4)
# inspect results
summary(galaxyRC)
str(galaxyRC)
# make galaxysentiment a factor [not ordered]
galaxyRC$galaxysentiment <- as.factor(galaxyRC$galaxysentiment)

# create a new dataset that will be used for recoding sentiment incorporating RFE data
#1: negative
#2: somewhat negative
#3: somewhat positive
#4: positive
galaxyRC_RFE <- galaxyRFE
galaxyRC_RFE$galaxysentiment <- as.numeric(galaxyRC_RFE$galaxysentiment)
# recode sentiment to combine factor levels 1 & 2 and 5 & 6 
galaxyRC_RFE$galaxysentiment <- recode(galaxyRC_RFE$galaxysentiment, '1' = 1, '2' = 1, '3' = 2, '4' = 3, '5' = 4, '6' = 4)
# inspect results
summary(galaxyRC_RFE)
str(galaxyRC_RFE)
# make galaxysentiment a factor [not ordered]
galaxyRC_RFE$galaxysentiment <- as.factor(galaxyRC_RFE$galaxysentiment)

#--- Dataset 2 ---#


##################
# Train/test sets
##################

# set random seed
set.seed(998)
# create the training partition that is 70% of total obs
inTraining_all <-
  createDataPartition(galaxymatrix$galaxysentiment, p = 0.70, list = FALSE)
# create training/testing dataset
trainSet_all <- galaxymatrix[inTraining_all, ]
testSet_all <- galaxymatrix[-inTraining_all, ]
# verify number of obs
nrow(trainSet_all) # 9040
nrow(testSet_all)  # 3871
str(trainSet_all$galaxysentiment)

#create train/test sets with no zero var data set
# set random seed
set.seed(998)
# create the training partition that is 70% of total obs
inTraining_nozv <-
  createDataPartition(galaxyNZV$galaxysentiment, p = 0.70, list = FALSE)
# create training/testing dataset
trainSet_nozv <- galaxyNZV[inTraining_nozv, ]
testSet_nozv <- galaxyNZV[-inTraining_nozv, ]
# verify number of obs
nrow(trainSet_nozv) # 
nrow(testSet_nozv)  # 

#create train/test sets with filtered correlation data set
# set random seed
set.seed(123)
# create the training partition that is 70% of total obs
inTraining_COR <-
  createDataPartition(galaxyCOR$galaxysentiment, p = 0.70, list = FALSE)
# create training/testing dataset
trainSet_COR <- galaxyCOR[inTraining_COR, ]
testSet_COR <- galaxyCOR[-inTraining_COR, ]
# verify number of obs
nrow(trainSet_COR) # 
nrow(testSet_COR)  # 



#create train/test sets for RFE data set
# set random seed
set.seed(123)
# create the training partition that is 70% of total obs
inTraining_RFE <-
  createDataPartition(galaxyRFE$galaxysentiment, p = 0.70, list = FALSE)
# create training/testing dataset
trainSet_RFE <- galaxyRFE[inTraining_RFE, ]
testSet_RFE <- galaxyRFE[-inTraining_RFE, ]
# verify number of obs
nrow(trainSet_RFE) # 
nrow(testSet_RFE)  # 


#create train/test sets for new feature engineered set with 4 factors
# set random seed
set.seed(123)
# create the training partition that is 70% of total obs
inTraining_RC <-
  createDataPartition(galaxyRC$galaxysentiment, p = 0.70, list = FALSE)
# create training/testing dataset
trainSet_RC <- galaxyRC[inTraining_RC, ]
testSet_RC <- galaxyRC[-inTraining_RC, ]
# verify number of obs
nrow(trainSet_RC) # 
nrow(testSet_RC)  # 

#create train/test sets for new feature engineered set with 4 factors and RFE set
# set random seed
set.seed(123)
# create the training partition that is 70% of total obs
inTraining_RC_RFE <-
  createDataPartition(galaxyRC_RFE$galaxysentiment, p = 0.70, list = FALSE)
# create training/testing dataset
trainSet_RC_RFE <- galaxyRC_RFE[inTraining_RC_RFE, ]
testSet_RC_RFE <- galaxyRC_RFE[-inTraining_RC_RFE, ]
# verify number of obs
nrow(trainSet_RC_RFE) # 9039
nrow(testSet_RC_RFE)  # 3872

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
#PCA needed 10 components to capture 80 percent of the variance

# use predict to apply pca parameters, create training, exclude dependent
train.pca <- predict(preprocessParams, trainSet_all[,-59])

# add the dependent to training
train.pca$galaxysentiment <- trainSet_all$galaxysentiment

# use predict to apply pca parameters, create testing, exclude dependent
test.pca <- predict(preprocessParams, testSet_all[,-59])

# add the dependent to testing
test.pca$galaxysentiment <- testSet_all$galaxysentiment

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
  galaxysentiment ~ . ,
  data = trainSet_all,
  method = 'kknn',
  preProcess = c('center', 'scale'),
  trControl = fitControl,
  tuneLength = 3
)

KKNN1
#kmax  Accuracy   Kappa    
#5     0.6788759  0.4277484
#7     0.7266612  0.4837082
#9     0.7389432  0.4978100

KKNN2 <- train(
  galaxysentiment ~ . ,
  data = trainSet_all,
  method = 'kknn',
  preProcess = c('center', 'scale'),
  trControl = fitControl,
  tuneLength = 7
)

#kmax  Accuracy   Kappa    
#5    0.6529509  0.4086456
#7    0.6976406  0.4574371
#9    0.7095889  0.4715040
#11    0.7187681  0.4845791
#13    0.7226382  0.4902203
#15    0.7276187  0.4961091
#17    0.7597347  0.5212189

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were kmax = 17, distance = 2 and kernel = optimal.

KKNN3 <- train(
  galaxysentiment ~ . ,
  data = trainSet_COR,
  method = 'kknn',
  preProcess = c('center', 'scale'),
  trControl = fitControl,
  tuneLength = 7
)

#kmax  Accuracy   Kappa    
#17    0.7563032  0.5133344
#The final values used for the model were kmax = 17, distance = 2 and kernel = optimal

KKNN3 <- train(
  galaxysentiment ~ . ,
  data = trainSet_nozv,
  method = 'kknn',
  preProcess = c('center', 'scale'),
  trControl = fitControl,
  tuneLength = 7
)

#kmax  Accuracy   Kappa    
#5    0.7091842  0.4483729
#7    0.7355110  0.4776277
#9    0.7294271  0.4702836
#11    0.7463499  0.4896906
#13    0.7494459  0.4932045
#15    0.7505522  0.4945197
#17    0.7508839  0.4938424

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were kmax = 17, distance = 2 and kernel = optimal.

KKNN4 <- train(
  galaxysentiment ~ . ,
  data = trainSet_RFE,
  method = 'kknn',
  preProcess = c('center', 'scale'),
  trControl = fitControl,
  tuneLength = 7
)

#kmax  Accuracy   Kappa    
#13    0.7421590  0.4995729
#15    0.7493439  0.5075469
#17    0.7472428  0.5048206

KKNNRC <- train(
  galaxysentiment ~ . ,
  data = trainSet_RC,
  method = 'kknn',
  preProcess = c('center', 'scale'),
  trControl = fitControl,
  tuneLength = 7
)

#kmax  Accuracy   Kappa  
#17    0.8348280  0.5745167

KKNNpca <- train(
  galaxysentiment ~ . ,
  data = train.pca,
  method = 'kknn',
  preProcess = c('center', 'scale'),
  trControl = fitControl,
  tuneLength = 7
)

#kmax  Accuracy   Kappa  
#15    0.7565244  0.5169712

## ------- C5.0 ------- ##

set.seed(123)

C50Fit1 <-
  train(
    galaxysentiment  ~ .,
    data = trainSet_all,
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7689207  0.5366059

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 1, model = rules and winnow = FALSE.

#with pre-processing
C50Fit2 <-
  train(
    galaxysentiment  ~ .,
    data = trainSet_all,
    preProcess = c('center', 'scale'),
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7046904  0.4794272

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were trials = 1, model = rules and winnow = FALSE.

#****correlated features removed data set****
C50Fit3 <-
  train(
    galaxysentiment  ~ .,
    data = trainSet_COR,
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7653778  0.5291839

#near zero var features removed
C50Fit4 <-
  train(
    galaxysentiment  ~ .,
    data = trainSet_nozv,
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7544291  0.5013422


#RFE feature selection dataset
C50Fit5 <-
  train(
    galaxysentiment  ~ .,
    data = trainSet_RFE,
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#model  winnow  trials  Accuracy   Kappa    
#rules  FALSE    1      0.7671494  0.5322086

C50FitRC <-
  train(
    galaxysentiment  ~ .,
    data = trainSet_RC,
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#model  winnow  trials  Accuracy   Kappa    
#tree   FALSE    1      0.8422415  0.5908059


C50Fitpca<-
  train(
    galaxysentiment  ~ .,
    data = train.pca,
    method = 'C5.0',
    metric = 'Accuracy',
    trControl = fitControl,
    tuneLength = 5
  )

#kmax  Accuracy   Kappa   
#11    0.7464669  0.5055461

## ------- SVM ------- ##

set.seed(123)


SVMFit1 <-
  train(
    galaxysentiment  ~ .,
    data = trainSet_all,
    preProcess = c('center', 'scale'),
    method = 'svmLinear2',
    trControl = fitControl,
    tuneLength = 5
  )

#cost  Accuracy   Kappa    
#0.25  0.6996631  0.3689732
#0.50  0.6998851  0.3692780
#1.00  0.7007702  0.3712164
#2.00  0.7006594  0.3706759
#4.00  0.7005488  0.3705819

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was cost = 1.

SVMFit2 <-
  train(
    galaxysentiment  ~ .,
    data = trainSet_all,
    method = 'svmLinear2',
    trControl = fitControl,
    tuneLength = 5
  )

#cost  Accuracy   Kappa    
#0.25  0.6995569  0.3677070
#0.50  0.7007737  0.3706633
#1.00  0.7009946  0.3714337
#2.00  0.7011049  0.3712346
#4.00  0.7008842  0.3708354

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was cost = 2.

## ------- rf ------- ##
set.seed(123)

rfFit1 <-
  train(
    galaxysentiment ~ .,
    data = trainSet_all,
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.7048763  0.3526286
#20    0.7705786  0.5408339
#39    0.7655979  0.5355691
#58    0.7619485  0.5310307


#with pre-processing and higher tuneLength
rfFit2 <-
  train(
    galaxysentiment ~ .,
    data = trainSet_all,
    preProcess = c('center', 'scale'),
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.7061935  0.3572776
#20    0.7705727  0.5410903
#39    0.7638250  0.5323782
#58    0.7588465  0.5256973

#COR dataset
rfFit3 <-
  train(
    galaxysentiment ~ .,
    data = trainSet_COR,
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.6994484  0.3371952
#16    0.7658197  0.5321522
#30    0.7610639  0.5264310
#44    0.7554240  0.5184564

#near zero var dataset
rfFit4 <-
  train(
    galaxysentiment ~ .,
    data = trainSet_nozv,
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.7567503  0.5036123

#RFE dataset
rfFit5 <-
  train(
    galaxysentiment ~ .,
    data = trainSet_RFE,
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.7385058  0.4541673
#7    0.7650526  0.5319219
#12    0.7594101  0.5243741
#18    0.7546540  0.5177885

#model using new recoded dv
rfFitRC <-
  train(
    galaxysentiment ~ .,
    data = trainSet_RC,
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.7861495  0.3695132
#20    0.8435700  0.5929621
#39    0.8409149  0.5889882
#58    0.8377070  0.5822585

varImp(rfFitRC)
#variables are sorted by maximum importance across the classes
#only 20 most important variables shown (out of 58)

#               1       2       3     4
#googleandroid 17.9274 45.3915 100.000 93.29
#samsunggalaxy 28.7606 13.2932  89.510 92.20
#iphone        88.8616 50.0766  52.453 77.19
#iphonedisunc  11.4546  3.3142  43.351 74.36
#iphonedispos  12.0760  0.9079  49.227 67.90
#iphoneperpos  14.7420  8.4598  43.051 30.38
#iphonedisneg   8.2209  4.2813  33.173 31.48
#iphoneperunc   6.3859  2.1009  29.662 23.70
#iphoneperneg  11.0626  6.0696  27.978 23.11
#htcphone      24.6080 17.1134  25.644 20.77
#iphonecampos  14.3077  2.6841  25.526 22.96
#iphonecamneg  11.8659  3.4254  23.071 18.21
#htccampos     18.7021 10.5941  16.531 21.19
#iphonecamunc  10.8622  1.7499  20.445 18.73
#samsungdisneg  5.3646  9.3058   9.220 19.85
#sonyperpos     0.3685  8.2500   7.349 19.59
#samsungdispos  3.9316  8.4742  13.112 19.31
#iosperunc      8.0201  7.0212   9.862 16.88
#htcdisunc      9.1597  8.2500  16.118 10.16
#sonyxperia    15.9959 10.1840   8.907 14.87

rfFitPCA <-
  train(
    galaxysentiment ~ .,
    data = train.pca,
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.7564189  0.5160370
#9    0.7569721  0.5172433
#17    0.7557537  0.5149911
#25    0.7555320  0.5154156

rfFit6 <-
  train(
    galaxysentiment ~ .,
    data = trainSet_RC_RFE,
    method = "rf",
    importance = T,
    trControl = fitControl,
    tuneLength = 4
  )

#mtry  Accuracy   Kappa    
#2    0.8114837  0.4746568
#7    0.8435696  0.5947126

##---------- adaboost ------------##

adaFit1 <-
  train(
    galaxysentiment ~ .,
    data = trainSet_all,
    preProcess = c('center', 'scale'),
    method = "AdaBag",
    trControl = fitControl,
    tuneLength = 5
  )

# maxdepth  mfinal  Accuracy   Kappa   
#5          50     0.7394928  0.4534099

##--- Compare metrics ---##

ModelFitResults <- resamples(list(C50 = C50Fit1, SVM = SVMFit1, rf = rfFit1, KNN = KKNN2))
# output summary metrics for tuned models
summary(ModelFitResults)

#Models: C50, SVM, rf, KNN 
#Number of resamples: 10 

#Accuracy 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C50 0.7491713 0.7625101 0.7693584 0.7689207 0.7766122 0.7831858    0
#SVM 0.6917960 0.6977580 0.7016571 0.7007702 0.7031813 0.7076412    0
#rf  0.7563677 0.7628403 0.7683793 0.7705786 0.7719976 0.7960089    0
#KNN 0.7477876 0.7513141 0.7567655 0.7597347 0.7682724 0.7751938    0

#Kappa 
#         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C50 0.4827517 0.5241182 0.5359833 0.5366059 0.5524826 0.5753030    0
#SVM 0.3454931 0.3611226 0.3729595 0.3712164 0.3804633 0.3927568    0
#rf  0.4985435 0.5217212 0.5389410 0.5408339 0.5483201 0.6009205    0
#KNN 0.4914795 0.5049497 0.5165835 0.5212189 0.5354647 0.5577361    0

#top three models are C5.0, RF, KNN. Use diff datasets for each model type

ModelFitResults2 <- resamples(list(C50 = C50FitRC, rf = rfFitRC, knn=KKNNRC))
summary(ModelFitResults2)
#Models: C50, rf, knn 
#Number of resamples: 10 

#Accuracy 
#         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C50 0.8263274 0.8372485 0.8429204 0.8422415 0.8466652 0.8528761    0
#rf  0.8307522 0.8387721 0.8402435 0.8435700 0.8493161 0.8592018    0
#knn 0.8250277 0.8274804 0.8363736 0.8348280 0.8403434 0.8438538    0

#Kappa 
#         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C50 0.5515845 0.5758964 0.5881663 0.5908059 0.6055157 0.6242585    0
#rf  0.5507279 0.5796875 0.5876413 0.5929621 0.6075323 0.6364135    0
#knn 0.5452274 0.5622823 0.5720477 0.5745167 0.5924132 0.6054373    0

ModelFitResults3 <- resamples(list(C50 = C50Fit3, rf = rfFit3, knn=KKNN3))
summary(ModelFitResults3)
#Call:
#  summary.resamples(object = ModelFitResults3)

#Models: C50, rf, knn 
#Number of resamples: 10 

#Accuracy 
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C50 0.7522124 0.7627877 0.7678275 0.7660414 0.7712942 0.7765487    0
#rf  0.7558011 0.7627877 0.7665929 0.7658197 0.7684009 0.7754425    0
#knn 0.7093923 0.7473663 0.7582943 0.7493439 0.7629978 0.7688053    0

#Kappa 
#         Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
#C50 0.5068242 0.5200452 0.5288856 0.5308805 0.5460383 0.5519834    0
#rf  0.5063653 0.5247152 0.5307993 0.5321522 0.5428973 0.5578077    0
#knn 0.4477239 0.4932720 0.5198786 0.5075469 0.5307978 0.5392855    0


#--- Save/load top performing model ---#


# load and name model



############################
# Predict testSet/validation
############################


C50Pred <- predict(C50FitRC, testSet_RC)
postResample(C50Pred, testSet_RC$galaxysentiment)
#Accuracy     Kappa 
#0.8447831 0.5984791  

C50ConfusionMatrix <- confusionMatrix(C50Pred, testSet_RC$galaxysentiment)
C50ConfusionMatrix
#Confusion Matrix and Statistics

        #Reference
#Prediction    1    2    3    4
#         1  362    2    6   34
#         2    2   17    0    2
#         3    3    3  200   34
#         4  256  113  146 2692

#Overall Statistics

#Accuracy : 0.8448         
#95% CI : (0.833, 0.8561)
#No Information Rate : 0.7133         
#P-Value [Acc > NIR] : < 2.2e-16      

#Kappa : 0.5985         

#Mcnemar's Test P-Value : < 2.2e-16      

#Statistics by Class:

#                     Class: 1 Class: 2 Class: 3 Class: 4
#Sensitivity           0.58106 0.125926  0.56818   0.9747
#Specificity           0.98707 0.998930  0.98864   0.5360
#Pos Pred Value        0.89604 0.809524  0.83333   0.8394
#Neg Pred Value        0.92474 0.969359  0.95815   0.8947
#Prevalence            0.16090 0.034866  0.09091   0.7133
#Detection Rate        0.09349 0.004390  0.05165   0.6952
#Detection Prevalence  0.10434 0.005424  0.06198   0.8283
#Balanced Accuracy     0.78407 0.562428  0.77841   0.7553

rfPred <- predict(rfFitRC, testSet_RC)
postResample(rfPred, testSet_RC$galaxysentiment)
#Accuracy     Kappa 
#0.8502066 0.6110204 
summary(rfPred)

rfConfusionMatrix <- confusionMatrix(rfPred, testSet_RC$galaxysentiment)
rfConfusionMatrix
#Confusion Matrix and Statistics

#         Reference
#Prediction    1    2    3    4
#           1  367    4    4   30
#           2    1   17    0    3
#           3    3    2  202   24
#           4  252  112  146 2705

#Overall Statistics

#Accuracy : 0.8499          
#95% CI : (0.8383, 0.8611)
#No Information Rate : 0.7133          
#P-Value [Acc > NIR] : < 2.2e-16       

#Kappa : 0.6105          

#Mcnemar's Test P-Value : < 2.2e-16       

#Statistics by Class:

#                     Class: 1 Class: 2 Class: 3 Class: 4
#Sensitivity           0.58909 0.125926  0.57386   0.9794
#Specificity           0.98830 0.998930  0.99176   0.5405
#Pos Pred Value        0.90617 0.809524  0.87446   0.8414
#Neg Pred Value        0.92616 0.969359  0.95880   0.9132
#Prevalence            0.16090 0.034866  0.09091   0.7133
#Detection Rate        0.09478 0.004390  0.05217   0.6986
#Detection Prevalence  0.10460 0.005424  0.05966   0.8303
#Balanced Accuracy     0.78869 0.562428  0.78281   0.7600

C50Pred2 <- predict(C50Fit1, testSet_all)
postResample(C50Pred2, testSet_all$galaxysentiment)
#Accuracy     Kappa 
#0.7594937 0.5154550 

rfPred2 <- predict(rfFit1, testSet_all)
postResample(rfPred2, testSet_all$galaxysentiment)

C50Pred3 <- predict(C50Fit5, testSet_RFE)
postResample(C50Pred3, testSet_RFE$galaxysentiment)
#Accuracy     Kappa 
#0.7662103 0.5286897 

KKNNPred <- predict(KKNNRC, testSet_RC)
postResample(KKNNPred, testSet_RC$galaxysentiment)
#Accuracy     Kappa 
#0.8422004 0.5942926 

###############################
# Predict new data (Dataset 2)
###############################

rfPredfinalgalaxy <- predict(rfFitRC, galaxylargematrix)
rfPredFinal2 <- predict(C50Pred3, galaxylargematrixRFE)

#delete blank galaxy column
galaxylargematrix$galaxysentiment <- NULL
galaxylargematrix <- mutate(galaxylargematrix,rfPredfinalgalaxy)
colnames(galaxylargematrix)
galaxylargematrix <- galaxylargematrix  %>% rename(galaxysentiment = rfPredfinalgalaxy)

#save to file
write.csv(galaxylargematrix,file = "galaxyLargeMatrix.csv")

#add predictions to dataset and merge with iphone large matrix & remove non-RC predictions
iphonelargematrix <-
  read.csv(
    "iphoneLargeMatrix.csv",
    stringsAsFactors = FALSE,
    header = T
  )

iphonelargematrix <- mutate(iphonelargematrix,rfPredfinalgalaxy)
iphonelargematrix$iphonesentiment2 <- NULL
iphonelargematrix$X <- NULL
colnames(iphonelargematrix)
iphonelargematrix <- iphonelargematrix  %>% rename(galaxysentiment = rfPredfinalgalaxy)

summary(rfPredfinalgalaxy)
#1     2     3     4 
#12282  1042  1562 15146 

#save to file
write.csv(iphonelargematrix,file = "iphone_galaxyLargeMatrix.csv")

###############################
# Data Visualizations
###############################
#read back in file
iphone_galaxymatrix <-
  read.csv(
    "iphone_galaxyLargeMatrix.csv",
    stringsAsFactors = FALSE,
    header = T
  )
colnames(iphone_galaxymatrix)
iphone_galaxymatrix$X.1 <- NULL
iphone_galaxymatrix$X <- NULL


plot_ly(iphone_galaxymatrix, x= ~iphone_galaxymatrix$galaxysentiment, name = 'Galaxy Sentiment', type='histogram') %>%
  add_histogram(x = ~iphone_galaxymatrix$iphonesentiment, name = 'iPhone Sentiment') %>%
  layout(title = "Smartphone Sentiment",
         xaxis = list(title = "Sentiment Rating"),
         yaxis = list (title = "Count"))

#sum top 5 varImp for models
samsunggalaxy <- sum(iphone_galaxymatrix$samsunggalaxy)
iphone <- sum(iphone_galaxymatrix$iphone)
googleandroid <- sum(iphone_galaxymatrix$googleandroid)
iphonedispositive <- sum(iphone_galaxymatrix$iphonedispos)
iphonedisunclear <- sum(iphone_galaxymatrix$iphonedisunc)

Variables <- c("samsunggalaxy", "iphone", "googleandroid","iphonedispositive","iphonedisunclear")
Count <- c(samsunggalaxy, iphone, googleandroid, iphonedispositive, iphonedisunclear)
data <- data.frame(Variables, Count)

plot_ly(data, x = ~Variables, y = ~Count, type = 'bar', text= Count, textposition = 'auto', name = 'Top Variables') %>%
  layout(yaxis = list(title = 'Count of Mentions'))
