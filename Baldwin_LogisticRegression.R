## Dr. Clif Baldwin
## October 23, 2017; revised April 16, 2018
## Based on code from https://dernk.wordpress.com/2013/06/03/gradient-descent-for-logistic-regression-in-r/
## For Machine Learning

##Edited by: Nicholas Paisley

library(ggplot2)

#Activation
sigmoid = function(z){ #The sigmoid produces an output between 0 and 1 (the probability estimate)
  1 / (1 + exp(-z)) #z = input to the function, while exp or 'e' is base of natural log
}
#######NOT NEEDED IN THIS CODE#######
#cost function
cost = function(X, y, beta){ #Computing the test set error (Cross Entropy)?
  m = nrow(X) #storing the "X" rows into m signifying our sample number in the dataset
  hx = sigmoid(X %*% beta) #output to the sigmoid function
  (1/m) * (((-t(y) %*% log(hx)) - t(1-y) %*% log(1 - hx))) #Adjusting the estimates in the logistic regression equation such that the total log loss function over the whole dataset is minimized.
  #m = number of samples in dataset
  #hx = is the predicted probability outcome by applying the logistic regression equation (sigmoid)
  #y = is the outcome (dependent variable) which is either "0" or "1".
}

gradient = function(X, y, beta){
  m = nrow(X) #storing the number of rows in the "x" variable
  X = as.matrix(X) #converting the "x" variable to a matrix 
  hx = sigmoid(X %*% beta) #output of the sigmoid function
  (1/m) * (t(X) %*% (hx - y)) #The gradient function
}

logisticRegression <- function(X, y, maxiteration) { #defining the variables needed to use in the function to execute logisticRegression
  alpha = 0.001 #learning rate of the logistic regression
  X = cbind(rep(1,nrow(X)),X) #Creating and combining a set column and row (with X amounts) of 1's.
  beta <- matrix(rep(0, ncol(X)), nrow = ncol(X)) #Creating a NxN matrix of 0's for the initial beta.
  
  #Finding new betas and then putting them back into the equation to get the equation to learn. 
  for (i in 1:maxiteration){ #how many iterations we are doing (depends on maxiteration amount)
    beta = beta - alpha * gradient(X, y, beta) #Updating the beta every time after an iteration to help it learn
  }
  return(beta) #returns the beta back into the equation
}

logisticPrediction <- function(betas, newData){  
  X <- na.omit(newData) #Gets rid of any "NA" variables/rows
  X = cbind(rep(1,nrow(X)),X) #Creating and combining a set column and row (with X amounts) of 1's. This is for the intercept term (B0)
  X <- as.matrix(X) #Changing X into a matrix
  return(sigmoid(X %*% betas))
}

testLogistic <- function(train, test, threshold) {
  #glm is used to fit generalized linear models, specified by giving a symbolic description of the linear predictor and a description of the error distribution.
	glm1 = glm(Y~.,data=train,family=binomial) #[Y~: Include all main terms, family=binomial: Logistic (binary answer), data=train: The type of data we are using]
	#predict() = Predicts values based on linear model objects
	#predict(object, newdata, type)
	#object: glm1 
	#newdata=test: An optional data frame in which to look for variables with which to predict
	#type="response": tells R to output probabilities of the form P(Y = 1|X) (Conditional Probability?) , as opposed to other information such as the logit
	TestPrediction = predict(glm1, newdata=test, type="response") #Returns an array of predictions
	vals <- table(test$Y, TestPrediction > threshold) #Getting all the values from the test table (Y) and all the values in the TestPrediction that are above a 0.5. (Need the Y=1 for the conditional probabilty)
	# 3   0    TN  FN    <- creating the confusion matrix
	# 15 232   FP  TP
	accuracy = (vals[1]+vals[4])/sum(vals) #True Negative (TN) + True Positive (TP) / True Positive (TP) + False Positive (FP) + False Negative (FN) + True Negative (TN) : correctly predicted observation to the total observations.
	print(paste("The R accuracy of the computed data",accuracy, sep = " ")) #
	sensitivity = vals[4]/(vals[2]+vals[4]) # TP / (FN + TP) : correctly predicted positive observations to the total predicted positive observations  
	specificity = vals[1]/(vals[1]+vals[3]) # TN / (TN + FP) : correctly predicted negative observations to the total predicted negative observations
	print(paste("The R sensitivity of the computed data",sensitivity, sep = " ")) #
	print(paste("The R specificity of the computed data",specificity, sep = " ")) #
}

ROC <- function(train, test) { 
  glm1 = glm(Y~.,data=train,family=binomial)#[Y~: Include all main terms, family=binomial: Logistic (binary answer), data=train: The type of data we are using
  TestPrediction = predict(glm1, newdata=test, type="response") #refer to TestPrediction summary in "testLogistic"
  sensitivity = vector(mode = "numeric", length = 101) #Creating a numeric vector that has the length of 101. Labeled sensitivity. 
  falsepositives = vector(mode = "numeric", length = 101) #Creating a numeric vector that has the length of 101. Labeled falsepositives.
  specificity = vector(mode = "numeric", length = 101) 
  
  thresholds = seq(from = 0, to = 1, by = 0.01) #Creating a sequence that goes from 0 -> 1 in 0.01 increments (101 times)
  for(i in seq_along(thresholds)) {
    #seq_along : seq_along() generates a sequence the same length of the argument passed, and in the context of a for loop is used to more easily generate the index to iterate over. (https://community.rstudio.com/t/seq-along-function/39304)
    vals <- table(test$Y, TestPrediction > thresholds[i]) #Getting all the values from the test table (Y) and all the values in the TestPrediction that are above a 0.5. (Need the Y=1 for the coditional probabilty)
    #Creates confusion matrix
    # 3   0    TN  FN    <-  creating the confusion matrix
    # 15 232   FP  TP
    sensitivity[i] = vals[4]/(vals[2]+vals[4]) # TP / (FN + TP) : correctly predicted positive observations to the total predicted positive observations  
    falsepositives[i] = vals[3]/(vals[1]+vals[3]) # false positives, or 1 - specificity (FP / (TN + FP))
    specificity[i] = vals[1]/(vals[1]+vals[3]) #TN / (TN + FP) : correctly predicted negative observations to the total predicted negative observations
  }

  #creating ROC curve
  roc <- ggplot() + 
    geom_line(aes(falsepositives, sensitivity), colour="red") + #plotting falsepositives and sensitivity 
    geom_abline(slope = 1, intercept = 0, colour="blue") + #creates a slope line at (0,0) going up by 1.
    labs(title="ROC Curve", x= "1 - Specificity (FP)", y="Sensitivity (TP)") + #labels on the graphs
    geom_text(aes(falsepositives, sensitivity), label=ifelse(((thresholds * 100) %% 10 == 0),thresholds,''),nudge_x=0,nudge_y=0) #putting the numbers on the ROC curve. Signifies the area under the curve.
  #ifelse(logical_expression, a , b)
  #logical_expression: Indicates an input vector, which in turn will return the vector of the same size as output.
  #a: Executes when the logical_expression is TRUE.
  #b: Executes when the logical_expression is FALSE.
  
  #My added ROC curve with correct variables (However, my ROc curve negative and not positive)
  roc2 <- ggplot() + 
    geom_line(aes(specificity, sensitivity), colour="red") + #plotting specificity and sensitivity 
    geom_abline(slope = 1, intercept = 0, colour="blue") +
    labs(title="True ROC Curve", x= "Specifitity (TN rate)", y="Sensitivity (TP)") +
    geom_text(aes(falsepositives, sensitivity), label=ifelse(((thresholds * 100) %% 10 == 0),thresholds,''),nudge_x=0,nudge_y=0) 
  
  show(roc)
  show(roc2)
}

### Data Preparation ###
# The provided training data
data1 <- read.csv("LogisticData_1.csv", header = TRUE) #reading in the data from CSV

library(caTools) # To split the data - you can use any technique to split the data if you prefer
set.seed(88) #Used before any function with generates random numbers. Holds the train and test data in place (does not change them).
#The seed number you choose is the starting point used in the generation of a sequence of random numbers, which is why (provided you use the same pseudo-random number generator) you'll obtain the same results given the same seed number.
split = sample.split(data1$Y, SplitRatio = 0.75) #Split data from vector Y into two sets in predefined ratio while preserving relative ratios of different labels in Y. 
#Used to split the data used during classification into train and test subsets.
train = subset(data1, split == TRUE) #Getting the training data. Getting 75% of the data from the random split of the data. (TRUE)
test = subset(data1, split == FALSE) #Getting the test data. Getting 25% of the data from the random split (FALSE)
rm(split) #removing split after getting our test and training data 

          #R X c
X <- train[,-ncol(train)] #Using our training data and removing the last column from it (Y) to leave our X variables columns and rows with in it.
y <- train[,ncol(train)] #Using our training data and only grabbing our last column (Y). 

### End Data Preparation ###

maxiteration = 150000 #Number of iterations that is given for the logistic regression
betas <- logisticRegression(X, y, maxiteration) #Creating our betas using the logisticRegression definition using our defined x, y and matrix iterations

X <- test[,-ncol(test)]  #Using our test data and removing the last column from it (Y) to leave our X variables columns and rows with in it.
predictedTest <- logisticPrediction(betas, X) #Running logisticPredection with the outputted betas and X (test) to predict where the test data goes.

threshold = 0.5 #Creates a reference point to classify the data (If high/low it classifies it differently)
vals <- table(test$Y, predictedTest > threshold) #Creates a confusion matrix. 
accuracy = (vals[1]+vals[4])/sum(vals) #True Negative (TN) + True Positive (TP) / True Positive (TP) + False Positive (FP) + False Negative (FN) + True Negative (TN) : correctly predicted observation to the total observations.
print(paste("The accuracy of the computed data",accuracy, sep = " "))
sensitivity = vals[4]/(vals[2]+vals[4]) # TP / (FN + TP) : correctly predicted positive observations to the total predicted positive observations
specificity = vals[1]/(vals[1]+vals[3]) # TN / (TN + FP) : correctly predicted negative observations to the total predicted negative observations
print(paste("The sensitivity of the computed data",sensitivity, sep = " ")) #print sensitivity result
print(paste("The specificity of the computed data",specificity, sep = " ")) #print specificity result

testLogistic(train, test, threshold) #Using testLogistic function (comparing glm vs code by scratch)

ROC(train, test) #Using ROC function

#-------------------------------------------------------------------------------------------------------------------
