rm(list = ls())
setwd("C:/edwisor/project")
getwd()
df= read.csv("day.csv")
df1=read.csv("day.csv")
#convert date column into datetime format to extract days
df1$dteday=as.POSIXlt(df1$dteday)

# deleting unnecessary columns
df1=subset(df1,select=-c(instant,casual,registered))

#Rename Columns for better understanding
colnames(df1)[c(1,3,4,8,11,13)] = c("date","year","month","weathercondition","humidity","totalcount")
#convert required variables to categories

cols=c("month","year","season","weekday","weathercondition")
df1[cols]=lapply(df1[cols], factor)
df1$workingday =as.factor(df1$workingday)
df1$holiday =as.factor(df1$holiday)

#Exploratory analysis

min(df1$totalcount)  #minimum count is 22
mean(df1$totalcount)  #mean count is 4504
median(df1$totalcount) #median count is 4548
max(df1$totalcount)   # maximum count is 8714


summary(df1[df1$season == 1,])  #for season 1
# minimum count is 431
#Median count:2209
#Mean count :2604 
#Max count  :7836
summary(df1[df1$season == 2,])  #for season 2
#minimum count : 795
#Median count:4942 
#Mean count :4992 
#Max count  :8362
summary(df1[df1$season == 3,])  #for season 3
#minimum count : 1115
#Median count:5354
#Mean count :5644 
#Max count  :8714
summary(df1[df1$season == 4,])   #for season 4
#minimum count : 22
#Median count:4634
#Mean count :4728 
#Max count  :8555

#holiday analysis
summary(df1[df1$holiday == 1,])  
#minimum count during holidays: 1000
#Median count during holidays:3351
#Mean count during holidays:3735 
#Max count during holidays :7403

#working day analysis
summary(df1[df1$workingday == 1,]) 
#minimum count during workingdays: 22
#Median count during workingdays:4582
#Mean count during workingdays:4585 
#Max count during workingdays :8362

#Visualizations
library(ggplot2)
library(corrgram)
library(dplyr)
install.packages("DMwR")
library(DMwR)

#1. outlier analysis using boxplots
x=colnames(select_if(df1,is.numeric))
par(mfrow= c(2,3))
#Create boxplots
for(i in x) {
  boxplot(df1[,i],xlab=i,col="red") 
}
#Outliers are present in "Humidity" and "windspeed" variables.we will remove these outliers
df1$windspeed =replace(df1$windspeed,df1$windspeed %in% boxplot.stats(df1$windspeed)$out,NA)
boxplot.stats(df1$windspeed)$out
df1$humidity =replace(df1$humidity,df1$humidity %in% boxplot.stats(df1$humidity)$out,NA)
df1$windspeed[4] =NA #=0.160296

mean(df1$windspeed,na.rm = TRUE) #0.1853633
median(df1$windspeed,na.rm = TRUE)#0.178483
df1[c(9,10,11,12)]=knnImputation(data=df1[c(9,10,11,12)],k=3)#0.1584833

df1$windspeed[4]
sum(is.na(df1))

#corellation analysis
install.packages("corrplot")
library(corrplot)
par(mfrow= c(1,1))
corrplot(cor(df1[c(x)]))
#atemp and temp have high correlation.to avoid multicollinearity we will remove atemp
df1=subset(df1,select =-c(atemp))
#temp vs tottalcount
ggplot(df1,aes(x=temp,y=totalcount))+geom_point(size=I(3))+geom_smooth(method = lm)
#there is a decent correlation between temperature and totalcount.The count tends to go up during warmer temperatures.
ggplot(df1,aes(x=season,y=totalcount)) +geom_bar(stat = "identity",fill="blue")+scale_x_discrete(labels=c("1"="spring","2"="summer","3"="fall","4"="winter"))
ggplot(df1,aes(x=humidity,y=totalcount))+geom_point(size=I(3))+geom_smooth(method = lm)
ggplot(df1,aes(x=windspeed,y=totalcount))+geom_point(size=I(3))+geom_smooth(method = lm)

library(Metrics)
#train test split
library(caTools)
colnames(df1)
#holiday and workingday tell the same exact thing.therefore we will exclude one of the variable.
df1=subset(df1,select =-c(holiday,date))
dim(df1)
set.seed(123)
split=sample.split(df1$totalcount,SplitRatio = 0.8)
trainset=subset(df1,split==TRUE)
testset=subset(df1,split==FALSE)
ytest=testset["totalcount"]

#linear regression
regressor=lm(formula = totalcount ~ season+year+temp+windspeed+humidity+weathercondition+workingday,data = trainset)
summary(regressor)
#weekday and month are showing low significance with p value more than 0.05.this is evident with the increase in accuracy of the model without these predictors.
ypred=predict(regressor,newdata = testset)
ypred
m1=lm(ytest$totalcount ~ ypred)
ggplot(data=m1,aes(x=ytest$totalcount,y=resid(m1)))+geom_point(size=2)+geom_hline(yintercept = 0)
mae(ytest$totalcount,ypred)
rmse(ytest$totalcount,ypred)
mape(ytest$totalcount,ypred)

#decision tree
library(rpart)
d1=rpart(totalcount ~ season+year+humidity+windspeed+weathercondition+workingday,data=trainset,method="anova")
summary(d1)
dt_predict=predict(d1,newdata = testset)
dt_predict
m3=lm(ytest$totalcount ~ dt_predict)
ggplot(data=m3,aes(x=ytest$totalcount,y=resid(m3)))+geom_point(size=2)+geom_hline(yintercept = 0)
mape(ytest$totalcount,dt_predict)
rmse(ytest$totalcount,dt_predict)
mae(ytest$totalcount,dt_predict)


#random forest
library(randomForest)
rf=randomForest(formula= totalcount ~ season+year+humidity+windspeed+weathercondition+workingday+temp,data=trainset)
summary(rf)
rf_ypred=predict(rf,newdata = testset)
rf_ypred
mae(ytest$totalcount,rf_ypred)
mape(ytest$totalcount,rf_ypred)
rmse(ytest$totalcount,rf_ypred)
m2=lm(ytest$totalcount ~ rf_ypred)
ggplot(data=m2,aes(x=ytest$totalcount,y=resid(m2)))+geom_point(size=2)+geom_hline(yintercept = 0)


#Random forest is giving best results so we will select random forest alogorith for our model


#sample input and output
predict(rf,newdata=testset[c(1,2),])

#output-  1634.043,1821.283




