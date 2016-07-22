# Practical Machine Learning Prediction Assignment Writeup
MUN YEE LEE  

##Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

#Data Processing
The training data for this project are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

Prepare the datasets by reading the training data and testing data into a data table.

```r
pmltrain <- read.csv('pml-training.csv')
pmltest <- read.csv('pml-testing.csv')
```

#Exploratory Data Analysis
###Create training,test and validation sets
Set all the library that required for analysis.

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.5
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.2.5
```

```r
library(ggplot2)
library(lattice)
library(kernlab)
```

```
## Warning: package 'kernlab' was built under R version 3.2.4
```

```
## 
## Attaching package: 'kernlab'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     alpha
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.2.5
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```


```r
#str(pmltrain)
#str(pmltest)

trainidx <- createDataPartition(pmltrain$classe,p=.9,list=FALSE)
traindata = pmltrain[trainidx,]
testdata = pmltrain[-trainidx,]
set.seed(32768)
nzv <- nearZeroVar(traindata)
trainnzv <- traindata[-nzv]
testnzv <- testdata[-nzv]
pmltestnzv <- pmltest[-nzv]

dim(trainnzv)
```

```
## [1] 17662   103
```

```r
dim(testnzv)
```

```
## [1] 1960  103
```

```r
dim(pmltestnzv)
```

```
## [1]  20 103
```

```r
ftridx <- which(lapply(trainnzv,class) %in% c('numeric'))
trainnzv1 <- preProcess(trainnzv[,ftridx], method=c('knnImpute'))

ftridx
```

```
##  [1]   7   8   9  11  13  15  17  18  19  20  21  22  23  24  25  26  27
## [18]  28  29  36  37  38  40  41  42  43  50  52  54  56  57  58  59  60
## [35]  61  62  63  64  66  67  68  69  70  71  72  73  74  75  76  77  78
## [52]  84  85  86  87  88  89  90  91  93  94  95  96 101 102
```

```r
trainnzv1
```

```
## Created from 360 samples and 65 variables
## 
## Pre-processing:
##   - centered (65)
##   - ignored (0)
##   - 5 nearest neighbor imputation (65)
##   - scaled (65)
```

```r
pred1 <- predict(trainnzv1, trainnzv[,ftridx])
predtrain <- cbind(trainnzv$classe,pred1)
names(predtrain)[1] <- 'classe'
predtrain[is.na(predtrain)] <- 0

pred2 <- predict(trainnzv1, testnzv[,ftridx])
predtest <- cbind(testnzv$classe, pred2)
names(predtest)[1] <- 'classe'
predtest[is.na(predtest)] <- 0

predpmltest <- predict(trainnzv1,pmltestnzv[,ftridx] )

dim(predtrain)
```

```
## [1] 17662    66
```

```r
dim(predtest)
```

```
## [1] 1960   66
```

```r
dim(predpmltest)
```

```
## [1] 20 65
```

```r
#dim(trainnzv1)
#str(trainnzv1)
```

#Modeling

```r
#mod1 <- train(classe ~ ., method="glm",data=predtrain)
#mod2 <- train(classe ~ ., method="rf",data=predtrain,trControl=trainControl(method="cv"),number=3)
#mod1 <- glm(classe ~ .,predtrain)
#mod1 <- glm.fit(classe ~ .,predtrain)
model <- randomForest(classe~.,data=predtrain)

predtrain1 <- predict(model, predtrain) 
print(table(predtrain1, predtrain$classe))
```

```
##           
## predtrain1    A    B    C    D    E
##          A 5022    0    0    0    0
##          B    0 3418    0    0    0
##          C    0    0 3080    0    0
##          D    0    0    0 2895    0
##          E    0    0    0    0 3247
```

```r
training <- as.data.frame(table(predtrain1, predtrain$classe))
#qplot(training)

predtest1 <- predict(model, predtest) 
print(table(predtest1, predtest$classe))
```

```
##          
## predtest1   A   B   C   D   E
##         A 554   4   0   2   0
##         B   2 374   8   2   0
##         C   0   1 333   3   0
##         D   1   0   1 314   1
##         E   1   0   0   0 359
```

```r
#qplot(table(predtest1, predtest$classe))

str(predpmltest)
```

```
## 'data.frame':	20 obs. of  65 variables:
##  $ roll_belt               : num  0.936 -1.007 -1.009 0.968 -1.002 ...
##  $ pitch_belt              : num  1.1988 0.2055 0.0686 -1.8802 0.1364 ...
##  $ yaw_belt                : num  0.0706 -0.8139 -0.8097 1.8234 -0.8107 ...
##  $ max_roll_belt           : num  0.0422 -0.8569 -0.8563 1.8448 -0.845 ...
##  $ min_roll_belt           : num  0.0714 -0.8278 -0.8271 1.881 -0.8308 ...
##  $ amplitude_roll_belt     : num  -0.099 -0.1354 -0.1354 -0.0519 -0.0832 ...
##  $ var_total_accel_belt    : num  -0.333 -0.307 -0.352 -0.371 -0.215 ...
##  $ avg_roll_belt           : num  0.86 -1.064 -1.07 0.896 -1.049 ...
##  $ stddev_roll_belt        : num  -0.451 -0.451 -0.442 -0.374 -0.254 ...
##  $ var_roll_belt           : num  -0.328 -0.33 -0.329 -0.322 -0.258 ...
##  $ avg_pitch_belt          : num  1.096 0.171 0.165 -1.907 0.119 ...
##  $ stddev_pitch_belt       : num  -0.6001 0.127 -0.0902 -0.4316 0.7478 ...
##  $ var_pitch_belt          : num  -0.389 -0.125 -0.257 -0.324 1.111 ...
##  $ avg_yaw_belt            : num  0.0608 -0.8456 -0.8449 1.8744 -0.8439 ...
##  $ stddev_yaw_belt         : num  -0.0952 -0.1176 -0.1195 -0.0682 -0.0755 ...
##  $ var_yaw_belt            : num  -0.0687 -0.0688 -0.0688 -0.0684 -0.0681 ...
##  $ gyros_belt_x            : num  -2.402 -0.266 0.268 0.559 0.171 ...
##  $ gyros_belt_y            : num  -0.762 -0.762 -0.25 0.904 -0.25 ...
##  $ gyros_belt_z            : num  -1.37 0.25 0.665 -0.124 0.54 ...
##  $ roll_arm                : num  0.319 -0.241 -0.241 -1.741 0.806 ...
##  $ pitch_arm               : num  -0.761 0.15 0.15 1.951 0.24 ...
##  $ yaw_arm                 : num  2.5076 0.0143 0.0143 -1.9747 1.443 ...
##  $ var_accel_arm           : num  0.3393 -0.4705 0.0304 -0.4291 -0.7393 ...
##  $ gyros_arm_x             : num  -0.8471 -0.6062 1.0355 0.0917 -1.0028 ...
##  $ gyros_arm_y             : num  0.865 1.3 -1.3 -0.3 1.229 ...
##  $ gyros_arm_z             : num  -0.812 -1.264 1.558 1.178 -1.463 ...
##  $ max_picth_arm           : num  1.2 -0.5 -0.5 -1.236 0.484 ...
##  $ min_roll_arm            : num  -0.373 0.726 0.726 0.751 0.209 ...
##  $ amplitude_pitch_arm     : num  0.0911 -1.0385 -1.0385 -0.1121 -0.3749 ...
##  $ roll_dumbbell           : num  -0.593 0.438 0.475 0.275 -1.787 ...
##  $ pitch_dumbbell          : num  0.97 -1.154 -1.091 -0.515 -1.147 ...
##  $ yaw_dumbbell            : num  1.518 -0.933 -0.93 -1.271 -0.188 ...
##  $ max_roll_dumbbell       : num  0.735 -1.069 -1.026 -0.809 -0.54 ...
##  $ max_picth_dumbbell      : num  1.154 -1.053 -0.963 -1.309 -0.22 ...
##  $ min_roll_dumbbell       : num  1.506 -0.493 -0.593 0.144 -0.404 ...
##  $ min_pitch_dumbbell      : num  1.8414 -0.7164 -0.6875 -0.9861 -0.0886 ...
##  $ amplitude_roll_dumbbell : num  -0.299 -0.628 -0.529 -0.8 -0.22 ...
##  $ amplitude_pitch_dumbbell: num  -0.444 -0.688 -0.593 -0.747 -0.213 ...
##  $ var_accel_dumbbell      : num  -0.231 -0.205 -0.186 -0.279 -0.279 ...
##  $ avg_roll_dumbbell       : num  -0.369 0.604 0.656 0.416 -0.911 ...
##  $ stddev_roll_dumbbell    : num  -0.344 -0.549 -0.5 -0.634 -0.307 ...
##  $ var_roll_dumbbell       : num  -0.376 -0.409 -0.4 -0.428 -0.315 ...
##  $ avg_pitch_dumbbell      : num  1.333 -1.087 -1.076 -0.556 -0.78 ...
##  $ stddev_pitch_dumbbell   : num  -0.362 -0.571 -0.592 -0.801 -0.026 ...
##  $ var_pitch_dumbbell      : num  -0.4147 -0.4499 -0.4744 -0.5099 -0.0784 ...
##  $ avg_yaw_dumbbell        : num  1.596 -0.945 -0.913 -1.252 -0.182 ...
##  $ stddev_yaw_dumbbell     : num  -0.471 -0.668 -0.598 -0.732 -0.226 ...
##  $ var_yaw_dumbbell        : num  -0.422 -0.458 -0.438 -0.469 -0.268 ...
##  $ gyros_dumbbell_x        : num  0.3025 0.1132 0.1447 -0.0383 0.0816 ...
##  $ gyros_dumbbell_y        : num  0.0224 0.0063 0.1507 -0.106 -0.8282 ...
##  $ gyros_dumbbell_z        : num  -0.2 -0.2416 -0.0879 0.0741 -0.1377 ...
##  $ magnet_dumbbell_z       : num  -0.7315 -0.5886 -0.0385 0.0473 1.8977 ...
##  $ roll_forearm            : num  0.99 0.694 0.898 -0.316 -1.946 ...
##  $ pitch_forearm           : num  1.374 -0.999 -1.531 -0.375 -0.452 ...
##  $ yaw_forearm             : num  1.322 0.838 0.712 -0.189 -0.653 ...
##  $ max_picth_forearm       : num  0.648 0.646 0.42 -0.884 0.257 ...
##  $ min_pitch_forearm       : num  0.792 0.666 0.76 0.486 -0.526 ...
##  $ amplitude_roll_forearm  : num  1.1223 0.5504 0.1797 -0.9405 -0.0963 ...
##  $ amplitude_pitch_forearm : num  -0.1832 -0.0893 -0.3054 -0.932 0.5608 ...
##  $ var_accel_forearm       : num  -0.615 -0.182 -0.751 -0.717 0.699 ...
##  $ gyros_forearm_x         : num  0.8941 1.4798 0.0311 1.8804 -1.4021 ...
##  $ gyros_forearm_y         : num  -1.068 -0.892 -0.268 0.196 0.953 ...
##  $ gyros_forearm_z         : num  -0.402 -0.179 0.071 0.898 0.354 ...
##  $ magnet_forearm_y        : num  0.0749 0.8055 0.6229 0.7898 -2.2937 ...
##  $ magnet_forearm_z        : num  0.605 1.299 1.055 0.345 -0.821 ...
```

#Result
The predictions for all the models test are show as below.

```r
predanswers <- predict(model, predpmltest) 
predanswers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
