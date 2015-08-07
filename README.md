---
title: "Tracking Exercises using Personal Activity Trackers"
author: "Karim Lalani"
date: "August 5, 2015"
output: html_document
---

## Overview

This document examines data collected from personal activity tracking devices from 6 participants and uses it to predict if exercises were done in correct manner.

### Notes from the Author
1. In order to maintain reproducibility of the report, the random number generator seed has been set (see below).
2. Since the file download takes a few seconds, logic is put in place to only download the files once, and not download them if they already exist in the workspace.
3. Since the number of observations and features is very large, the model fit takes a long time
    a. In order to help with that, parallelization feature in R is leveraged. It is advised that not all cores be used as it may make the system unstable and unresponsive for other tasks. In the case of this report, only 6 out of 8 availabe cores were used.
    b. Since the model fit takes a long time, the model fit objects are saved to disk in the workspace on first run and loaded from disk for every subsequent generation of this document. This proved more efficient than caching results for chuncks as caches are lost if R code chuncks are modified.


```r
set.seed(1234)
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(gbm))
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(splines))
suppressPackageStartupMessages(library(plyr))
suppressPackageStartupMessages(library(foreach))
suppressPackageStartupMessages(library(parallel))
suppressPackageStartupMessages(library(doParallel))
suppressPackageStartupMessages(library(iterators))
registerDoParallel(cores=6)
```

## Getting the Data
Data is made available from these links: [Training data] [1], [Assignment data] [2]


```r
if(!file.exists("pml-training.csv")) {
  download.file(
    "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
    destfile = "pml-training.csv",method="curl")
}
if(!file.exists("pml-testing.csv")) {
  download.file(
    "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
    destfile = "pml-testing.csv",method="curl")
}
pmldata <- read.csv("pml-training.csv")
assignment <- read.csv("pml-testing.csv")

pmldim <- dim(pmldata)
quizdim <- dim(assignment)
```

The pml data contains ``19622`` observations and ``160`` features.

The assignment data contains ``20`` observations and ``160`` features.

## Cleaning and tidying the data
When the data was read in the first time, it was observed that the not all data types were correctly inferred due to the presence of missing values. The data contained many columns where data was not available for all observations.
To overcome this, all the columns where it made sense to have numeric data were coerced to be numeric.


```r
pmldata[,8:159] <-sapply(pmldata[,8:159],as.numeric)
```

## Data slicing
The data would be sliced 70/30 into training and testing sets. The cross validation set would come from 25% of the training set as 10 k-folds. While the train/test data slicing is done in code, the cross validation sample slicing would be handled by the "trControl" parameter of the "train" command.


```r
inTrain <- createDataPartition(y = pmldata$classe,p=0.7,list=FALSE)
training <- pmldata[inTrain,]
testing <- pmldata[-inTrain,]
ctrl <- trainControl(method="cv", savePredictions = TRUE,returnData = TRUE)
```

## Feature selection
It was observed that many variables in the data were missing for most of the observations. Also, where were a few variables that were missing for a few observations. In order to simplify the model, variables with missing values were removed from both the training and assignment data rather than imputing them.


```r
valcolstrain <- as.vector(sapply(pmldata,function(x){!any(is.na(x))}))
valcolstest <- as.vector(sapply(assignment,function(x){!any(is.na(x))}))
training <- training[,valcolstrain & valcolstest]
testing <- testing[,valcolstrain & valcolstest]
quizdata <- assignment[,valcolstrain & valcolstest]
```

There are ``67`` features in pml data and ``100`` features in the assignment data with null values.
After eliminating them the features with null values from both data sets, we are left with ``60`` non null features.

First seven features appear to be metadata about the observation. Those are eliminated as well.


```r
training <- training[,-(1:7)]
testing <- testing[,-(1:7)]
quizdata <- quizdata[,-(1:7)]
```

After the elimination, ``52`` features remain in the datasets excluding the outcome feature, "classe".

## Pre Processing
Since the dataset contains many features that appear similar or related, such as max, min, total etc on the same readings, Principal Component Analysis is done on the dataset to reduce the features while retaining 95% (default for "thresh" argument) of the variability in data.


```r
preProc <- preProcess(training[,-53],method="pca")
trainPC <- predict(preProc,training[,-53])
testPC <- predict(preProc,testing[,-53])
quizPC <- predict(preProc,quizdata[,-53])
```

After preprocessing with pca, centering, and scaling (default behavior), ``25`` principal components remain in the resulting dataset, that capture ``95``% variability of the data.

## Model selection
Since this is a classification problem, Random Forest and Generalized Boosted Regression models were chosen. The models would be fit to the data and accuracy would be compared to help identify the best fitting model. Furthermore, both models would also be stacked together to see if any additional accuracies could be gained.


```r
if(file.exists("gbmFit.RData")) {
  load("gbmFit.RData")
} else {
  gbmFit <- train(training$classe ~ .,data=trainPC,method="gbm",trControl = ctrl,verbose=FALSE)
  save(gbmFit,file="gbmFit.RData")
}
if(file.exists("rfFit.RData")) {
  load("rfFit.RData")
} else {
  rfFit <- train(training$classe ~ .,data=trainPC,method="rf",trControl=ctrl, verbose=FALSE)
  save(rfFit,file="rfFit.RData")
}
```

## Accuracy and Errors
It is expected that the in-sample error rate would be low  and the prediction accuracy would be high given the models that are selected. Cross Validation Sampling will help with identifying the right model pick by providing an estimate of out-of-sample error and prediction accuracy. This will allow keeping the test data set out of the model selection process and hence it won't influence the process.

It is estimated that the out-of-sample error will be higher than the in-sample-error.


```r
predGbmIn <- predict(gbmFit,trainPC)
predRfIn <- predict(rfFit,trainPC)
cmGbmIn <- confusionMatrix(training$classe,predGbmIn)
cmRfIn <- confusionMatrix(training$classe,predRfIn)
accGbmIn <- cmGbmIn$overall[1]
accRfIn <- cmRfIn$overall[1]  
accGbmOut <- sum(gbmFit$pred$obs == gbmFit$pred$pred)/nrow(gbmFit$pred)
accRfOut <- sum(rfFit$pred$obs == rfFit$pred$pred)/nrow(rfFit$pred)
```

|Model                    |In Sample Accuracy|In Sample Error Rate|Out of Sample Accuracy|Out of Sample Error Rate|
|-------------------------|------------------|--------------------|----------------------|------------------------|
|Generalized Boosted Model|``0.8629249``  |``0.1370751``|``0.6959469``     |``0.3040531``   |
|Random Forest            |``1``   |``0`` |``0.9685036``      |``0.0314964``    |

### In Sample Sensitivity and Specificity 
Generalized Boosted Model

```r
cmGbmIn$byClass[,1:2]
```

```
##          Sensitivity Specificity
## Class: A   0.8874237   0.9717901
## Class: B   0.8542251   0.9532918
## Class: C   0.7608536   0.9702332
## Class: D   0.8704639   0.9691176
## Class: E   0.9440000   0.9650910
```
Accuracy: ``0.8629249``
Error Rate: ``0.1370751``

Random Forest Model

```r
cmRfIn$byClass[,1:2]
```

```
##          Sensitivity Specificity
## Class: A           1           1
## Class: B           1           1
## Class: C           1           1
## Class: D           1           1
## Class: E           1           1
```
Accuracy: ``1``
Error Rate: ``0``

## Final model selection
From observing the accuracies and estimated error rates of the both model fits, it is clear that Random Forest Modeling provided a much better model than Generalized Boosted Modeling. 


```r
predGbm <- predict(gbmFit,testPC)
predRf <- predict(rfFit,testPC)
accGbmTest <- confusionMatrix(testing$classe,predGbm)$overall[1]
accRfTest <- confusionMatrix(testing$classe,predRf)$overall[1]
predDF <- data.frame(predRf,predGbm,classe = testing$classe)
combModFit <- train(classe ~ .,predDF,methpd="gbm")
combPred <- predict(combModFit,predDF)
accTest <- confusionMatrix(testing$classe,combPred)$overall[1]
quizPred <- predict(rfFit,quizPC)
```

|Model                    |In Sample Accuracy|In Sample Error Rate|Out of Sample Accuracy|Out of Sample Error Rate|
|-------------------------|------------------|--------------------|----------------------|------------------------|
|Generalized Boosted Model|``0.8629249``  |``0.1370751``|``0.8231096``    |``0.1768904``  |
|Random Forest            |``1``   |``0``   |``0.9751912``     |``0.0248088``   |
|Stacked                  |                  |                    |``0.9751912``       |``0.0248088``     |

It is also observed that stacking Random Forest and GBM did not provide any additional increase in prediction accuracies. It is hence concluded that Random Forest Model fit will be used for prediction on the PML activity data.



[1]: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv "PML Training Data"
[2]: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv "PML Testing Data"
