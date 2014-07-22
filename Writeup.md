## Backgrownd

For this assignment, I build a predictive model to determine whether a particular form of exercise (barbell 

lifting) 
is performed correctly, using accelerometer data.

## Loading of the packages
The first part is the declaration of the package which will be used. 

Note : to be reproductible, I also set the seed value.

```{r}
library(caret)
library(randomForest)
library(Hmisc)
library(corrplot)

set.seed(1234)
```
## Reading of the Data

We begin by reading in the training and testing datasets, 
assigning missing values to entries that are currently 'NA' or blank.
(commands are commented to limit the output size. You can run it deleting the "#" ) 

```{r}
#getwd()
setwd("C:/PredictiveAnalysis/R")

train <- read.csv("./data/pml-training.csv", header = TRUE, na.strings = c("NA", ""))
test <- read.csv("./data/pml-testing.csv", header = TRUE, na.strings = c("NA", ""))
dim(test)
dim(train)
```

#names(train)

## Cleaning the data

We see that the training set consists of 19622 observations of 160 variables

```{r}
sum(complete.cases(train))
#head(train)
```

The discussion here is choosing between:
Or discarding most of the observations but using more predictors 
Or discarding some predictors to keep most of the observations.
The conclusion is that more observations is better, while additional variables may or may not helping us.


Columns in the orignal training and testing datasets that are mostly filled with missing values are removed. 

```{r}
# colSums(is.na(train))   ## This will give us which columns have 0 values
# colSums(is.na(train))==0  ## will give us a set of FALSE and TRUE values 

newtrain<- train[,colSums(is.na(train))==0]
newtest<- test[,colSums(is.na(test))==0]
dim(newtrain)
```

As we see we have eliminated 2/3 of the columns. 60 columns.

#head(newtrain)

Some of the variables in this new data set do not come from accelerometer measurements and record 

experimental
setup or participants' data.

So the following variables will be take out as well: 
X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window and num_window.

```{r}
todelete_cols <- grepl("X|user_name|new_window|num_window|raw_timestamp_part_1|raw_timestamp_part_2|

cvtd_timestamp", colnames(newtrain))
finaltrain <- newtrain[, !todelete_cols]
finaltest <- newtest[, !todelete_cols]
#names(finaltrain)
dim(finaltrain)

```
Now we have 53 columns to work with.

## Spliting the data for validation purposes

Now we will split the final training dataset into a training (70% of the observations) and a validation 
(30% of the observations). 

This validation dataset will allow us to perform cross validation for developing/testing our model.


```{r}
inTrain = createDataPartition(y = finaltrain$classe, p = 0.7, list = FALSE)
finaltrain_train = finaltrain[inTrain, ]
finaltrain_valid = finaltrain[-inTrain, ]
```

## Understanding the Correlations between variables

We begin by looking at the correlations between the variables in our dataset. 
We may want to remove highly correlated predictors from our analysis and replace them with weighted 
combinations of predictors. 

The goal of your project is to predict the manner in which they did the exercise. 
This is the "classe" variable in the training set. 

```{r}
correlMatrix <- cor(finaltrain_train[, -53])
corrplot(correlMatrix, order = "FPC", method = "circle", type = "lower", tl.cex = 0.8,  tl.col = rgb(0, 0, 

0))
```

This correlation plot shows the correlation between pairs of the predictors in our dataset. 
From a high-level perspective darker blue and darker red circles indicate high positive and high negative 

correlations, 
respectively. 

Nonetheless, there are a few pairs of variables that are highly correlated:

```{r}
which(correlMatrix > 0.98 & correlMatrix != 1)
correlMatrix[which(correlMatrix > 0.98 & correlMatrix != 1)]
correlMatrix[which(correlMatrix < -0.98 )]
```


## Predictive Model - Method 1
###############################
Now , its time to pre-process the data using a principal component analysis, leaving out the column we want 

to predict ('classe'). 
After pre-processing, we use the 'predict' function to apply the pre-processing to the training and 

validation 
of the final training dataset.

```{r}
preProc <- preProcess(finaltrain_train[, -53], method = "pca", thresh = 0.99)
trainPC <- predict(preProc, finaltrain_train[, -53])
validationTestPC <- predict(preProc, finaltrain_valid[, -53])
```

Now, we train a model using a random forest approach on the smaller training dataset. 
Note that we chose to specify the use of a cross validation method when applying the random forest routine in 

the
'trainControl()' parameter. Worth mentioning that without specifying this, the default method (bootstrapping) 
would have been used. The bootstrapping method take a much longer time to complete, and we get the same level 

of 'accuracy'.

Based on our problem: multi-dimensional classification with number of observations much exceeding the number 

of predictors,
Random forest is a good choice.

```{r}
modelFit <- train(finaltrain_train$classe ~ ., method = "rf", data = trainPC, trControl = trainControl(method 

= "cv", number = 4), importance = TRUE)
modelFit
modelFit$finalModel

```
This took some minutes to process, (and therefore I'll try using a different method.)
This method will give us an error rate: 1.88%

Now can review the relative importance of the resulting principal components of the trained model, 

'modelFit':
```{r}
varImpPlot(modelFit$finalModel, sort = TRUE, type = 1, pch = 19, col = 1, cex = .6,  main = "Importance of 

the Individual Principal Components")
```
Now we show the The degree of importance is shown on the x-axis–increasing from left to right. 


## Cross Validation Testing and Out-of-Sample Error Estimate
Call the 'predict' function again so that our trained model can be applied to our cross validation test 

dataset.

```{r}
predictionvalidationrf <- predict(modelFit, validationTestPC)
confusionMatrix(finaltrain_valid$classe, predictionvalidationrf)
```

Which gives an Accuracy : 97.7% 
And an estimated out-of-sample error based applying to the cross validation dataset is 2.33%.

## Predicted Results

```{r}
testPC<-predict(preProc, finaltest[,-53])
predictionfinal1<-predict(modelFit, testPC)
predictionfinal1
'''
Conclusion: The model achieves 90% accuracy on the testing set provided.
So, I am going to predict with a different function (randomForest)


## Predictive Model - Method 2
##############################
Using the randomForest model fuction with the complete dataset (before spliting it for the validation process 

used in the
Method 1)

For the dimensions of our problem I would start ntree=1024
```{r}
set.seed(1234)
dim(finaltrain_train)

model3 <- randomForest(classe ~ ., data = finaltrain, ntree = 1024)
model3
```

It will give us an OOB estimate of  error rate: 0.27%, which is very good!
As well the confusion Matrix looks good, indicating that the model fit the training set well.


Now lets calculate the variable importance order estimate that we got from the classifier training algorithm.

```{r}
imp3<-varImp(model3)
imp3$variables<-row.names(imp3)
imp3[order(imp3$Overall,decreasing=T),]
```
Only very few variables have lower importance measure more than the most important variables (roll_belt, 

yaw_belt), 
which seems to indicate the algorithm employed by them, it made good use of provided predictors.


## Predicted Results and apply it on 20 test cases for automatic grading
The following command can be used to obtain model's prediction for the assigned testing data set:

```{r}
finalprediction3<-predict(model3, finaltest)
'''

Conclusion: The model achieves 100% accuracy on the testing set provided.



Writing the text files, we'll use the following function:

```{r}
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

pml_write_files(as.character(finalprediction3))
```

## Making my PC to work in parallel to cut the processing time
Not in the scope of this assignment, it will be convenient to add the following statements to accelerate the 

processing
time by using in parallel more than one CPUs

```{r}
install.packages("doParallel")
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
```

