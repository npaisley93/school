## Dr. Clif Baldwin
## October 19, 2017
## For Machine Learning

##Commented by: Nicholas Paisley

#the library to be loaded in
library(ggplot2) 
library(standardize)
library(tidyverse)


# Creating a linearRegression variable as function(ds)
linearRegression <- function(ds) {
	# Creating the intercept variable that has a column vector of 1's with the rows of the data (i.e 4 rows of data = 4 rows of 1s). The row of 1's makes sure that the b0 is in the equation still and does not get taken out or changed.
	intercept <- rep(1, nrow(ds))
	# Creating the variable "ds" (updating) and adding the intercept matrix to the existing variable
	ds <- cbind(intercept, ds)
	
	
	# 
	vars <- length(ds) # Set the length of ds to the vars variable 
	     #rows x columns
	X <- ds[,1:vars-1] # All rows , removing the last column of vars. so we only have "x" values listed (i.e If length = 7 and saved as vars, 1:vars-1 = 6) 
	X <- as.matrix(X)  # Saves out X variable as a matrix value
	y <- ds[,vars]     # All rows , keeping all the "vars" variable and storing* it in a "y" variable

	# OLS (closed-form solution) #For linear regression on a model of the form y=XBeta, where X is a matrix with full column rank, the LSE is the following
	                             #Full column rank <- when each of the columns of the matrix are linearly independent and there are more rows than columns (m>n)
	XtXinv <- solve(t(X) %*% X) # solves the matrix for (X^TX)^-1
	XtXinvXt <- XtXinv %*% t(X) # solves the matrix for (X^TX)^-1X^T
	XtXinvXtY <-XtXinvXt %*% y  # solves the matrix for (X^TX)^-1X^Ty
	beta_hat <- XtXinvXtY       # (X^TX)^-1X^Ty = B^hat
	return(beta_hat) #stores beta_hat as the output vector for the function(ds) 
}

#Predicted value of y for a given value of x (fitted value)
predictY <- function(ds, beta_hat) { #
	X <- ds[,1:length(ds)-1]        #  Saving "x" variable as all rows and the length of ds - 1 (this takes out the Y variable)
	X <- cbind(rep(1, nrow(X)), X)  #  Adding a column of 1's to the first column and then adding all the x-variables after
	
	#[1,x11, ... x1p]
	#[1,x21, ... x2p]
	#[1,x31, ... x3p]
	#[1,x41, ... x4p]
	
	X <- as.matrix(X)               # Changing the "x" variable to a matrix 
	y_hat <- X %*% beta_hat         # Creating the predict value of y, which models the data set 
	return(y_hat)
}

#
errors <- function(y, y_hat) {
	RSS <- sum((y - y_hat)^2)  #Residual of Square Sum -> Depicts how the variation in the dependent variable in a regression model cannot be explained by the model. Generally, a lower residual sum of squares indicates that the regression model can better explain the data while a higher residual sum of squares indicates that the model poorly explains the data.
	TSS <- sum((y - mean(y_hat))^2) #Total Sum of Squares -> Measures the total variation of the sample. The variation of the values of a dependent variable (y_obs) from the sample mean of the dependent variable (in this case we are taking it from the y_model and not y_obs) It measures the overall difference between your data and the values predicted by your estimation model. Tells you how much variation there is in the dependent variable.
	R2 <- 1 - RSS/TSS  #R-squared -> The proportion of variables explained by the model, from 0-1. Measures provides information about the goodness of fit of a model. It is a statistical measure of how well the regression line approximates the actual data.
	RMSE <- sqrt(mean((y_hat - y)^2)) #Root Mean Squared Error -> Used for multiple linear regression. is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. 
	return(list(R2 = R2, RMSE = RMSE, RSS=RSS, TSS=TSS))  #Returns the values of the errors into a list under the errors in the global environment. 
	#mean() calculates the mean of the dataset. 
}

#
testlm <- function(train, test){ 
	model1 <- lm(data=train, Y~.) #Using the built in R lm() function (linear regression). lm(using the training data defined before , the columns used in Y
	predictTest = predict(model1, newdata=test) #Stores the predicted y-values ( which would be our y_model) as "predictTest". 
	                                            #newdata = an optional dataframe in which to look for variables with which to predict. If omitted, the fitted values are used. 
	
	#Plotting our y_obs values vs our y_model values created by the lm() function  
	plot(x=test$Y, y=predictTest, pch = "+", col='blue', main = "Actual Vs. Predicted by R") 
	
	#
	error <- errors(test$Y, predictTest) #Saves the answers from the "errors" function as "error"
	return(error$R2) #returns the Residual errors (Coefficent of determination) 
}

# Get the data
data1 <- read.csv("TrainData_Group1.csv", header = TRUE) #Reading in the data to "data1"

#sample takes a sample of the specified size from the elements of x using either with or without replacement.

indexes = sample(1:nrow(data1), size=0.75*nrow(data1)) #Grabs a 75% sample of the data from data1 and saves it in the variable "indexes" 

#Partitioning the data into training and testing data
train = data1[indexes,] # Grabbing all the rows of the "indexes" variable along with all of the columns associated with the "indexes" variable (75% of the data)
test = data1[-indexes,] # Grabbing all the rows of the of the data that was not included in the "indexes" variable along with all of the remaining columns not associated with the "indexes" column (25% of the data)
rm(indexes) # removes the "indexes" variable 

#Observed value of y 
beta_hat <- linearRegression(train) #saves beta_hat variable after going through the linearRegression function. (To observe the y observed values)

#Predicted value of y 
y_hat <- predictY(test, beta_hat) #saves y_hat variable after going thought the predictY function.(To predict the y_model values)

#Graphing the Actual Vs. Predicted using Algorithm
plot(x=test$Y, y=y_hat, pch = "+", col='red', main = "Actual Vs. Predicted using Algorithm") #Graphing the y-value of the data1 and the y value is the y_hat. Using "+" marks as plot symbols, in red.

# Residual errors (Coefficent of determination) from our formula (The closer to 1 the R2 is the better it fits the model)
#R-squared is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.

#0% indicates that the model explains none of the variability of the response data around its mean.
#100% indicates that the model explains all the variability of the response data around its mean.

error <- errors(test$Y, y_hat) #Using our equation the "errors" function 

#printing the R-squared from our algorithm vs R-squared from the lm() function
print(paste("R^2 from my algorithm",error$R2, sep = " ")) 

print(paste("R^2 from lm()",testlm(train, test), sep = " "))


#--------------------------------------------------------------------------

#Using lm to do linear regression with the data

df = subset(data1, select = c("X1","X2", "X3","X4","X5"))

x <- as.matrix(data1[-ncol(data1)]) #cretaes x matrix (removing the y_value)
y <- data1$Y #y data
summary(x)
summary(y)

int <- rep(1, length(y)) #creates rows of 1's the same size as the y-variable
summary(int)

x <- cbind(int,x) #combining int and x to put row of 1's in the matrix

#This is the equation for the beta_hat (y)
Beta_hat <- solve(t(x) %*% x) %*% t(x) %*% y

#Beta_hat <- round(Beta_hat,2)
print(Beta_hat)

# Linear regression model
lm.mod <- lm(y ~ ., data=df) #using lm to calculate the linear regression
print(lm.mod)

# Round for easier viewing
lm.B <- round(lm.mod$coefficients, 2) #round to 2 decimals
print("The rounded data is the following:") 
print(lm.B)

# Create data.frame of results
results <- data.frame(our.results=Beta_hat, lm.results=lm.B) #creating a frame to compare results side by side 

print(results) #printing results

plot(x=Beta_hat, y=lm.B) #plotting results against each other
