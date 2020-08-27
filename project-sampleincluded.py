#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[41]:


df= pd.read_csv("C:\edwisor\Project\day.csv")
df1= pd.read_csv("C:\edwisor\Project\day.csv")


# In[42]:


df.head(7)


# In[43]:


df1.rename(columns={"dteday":"date","yr":"year","mnth":"month","weathersit":"weathercondition","hum":"humidity","cnt":"totalcount"},inplace=True)


# In[44]:


df1["date"]=pd.to_datetime(df1["date"])
df1["season"]=pd.Categorical(df1["season"])
df1["weekday"]=pd.Categorical(df1["weekday"])
df1["holiday"]=pd.Categorical(df1["holiday"])
df1["weathercondition"]=pd.Categorical(df1["weathercondition"])
df1["workingday"]=pd.Categorical(df1["workingday"])
df1["month"]=pd.Categorical(df1["month"])
df1["year"]=pd.Categorical(df1["year"])


# In[45]:


df1.shape


# In[46]:


df1.info()


# In[47]:


df1.describe()


# Mean count over 2 years : 4504
# 
# 

# minimum count over 2 years: 22

# maximum count over 2 years: 8714

# In[48]:


df1.isna().sum()


# no missing values in the dataset

# In[49]:


df1["totalcount"].skew()


# skew value is close to zero. which means the distribution of totalcount is fairly symetrical although the negative skew indicates that most of the counts are more than the average count.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[50]:


(df1["totalcount"] <= 4504).sum()


# In[51]:


(df1["totalcount"] > 4504).sum()


# In[52]:


df1.corr()


# DATA VISUALIZATION

# In[53]:


sns.boxplot(df1["humidity"])


# In[54]:


sns.boxplot(df1["windspeed"])


# In[55]:


wind_hum=pd.DataFrame(df1,columns=['windspeed','humidity'])
cnames=["humidity","windspeed"]
for i in cnames:
    q75,q25=np.percentile(wind_hum.loc[:,i],[75,25])
    iqr=q75-q25
    min=q25-(iqr*1.5)
    max=q75+(iqr*1.5)
    wind_hum.loc[wind_hum.loc[:,i]<min,:i]=np.nan
    wind_hum.loc[wind_hum.loc[:,i]>max,:i]=np.nan
    


# In[56]:


wind_hum["windspeed"][3]=np.nan


# In[20]:


wind_hum['windspeed']=wind_hum['windspeed'].fillna(wind_hum['windspeed'].mean())
wind_hum["windspeed"][3]# mean=0.189


# In[39]:


wind_hum['windspeed']=wind_hum['windspeed'].fillna(wind_hum['windspeed'].median())
wind_hum["windspeed"][3] #median=0.180


# In[57]:


from sklearn.impute import KNNImputer 
imputer = KNNImputer(n_neighbors=3)
fit=imputer.fit_transform(wind_hum)


# In[58]:


fit[3][0]  #knnimpute=0.1225


# since median is the closest we will replace missing values with median

# In[62]:


wind_hum["windspeed"]=wind_hum["windspeed"].fillna(wind_hum["windspeed"].median())
wind_hum["humidity"]=wind_hum["humidity"].fillna(wind_hum["humidity"].median())


# In[63]:


df1["humidity"]=df1["humidity"].replace(wind_hum["humidity"])
df1["windspeed"]=df1["windspeed"].replace(wind_hum["windspeed"])


# Corellation analysis

# In[65]:


sns.heatmap(df1.corr(),annot= True)


# the heatmap says that temperature has the highest corellation with total count. to avoid multicollinearity we will remove "atemp" from the dataset as temp and atemp have high correlation.

# removing unnecessary variables from dataset

# In[66]:


df1=df1.drop(columns={"instant","atemp","casual","registered"})


# In[77]:


ax=sns.barplot(x=df1["season"],y=df1["totalcount"],hue=df1["holiday"])
ax.set_xticklabels(["spring","summer","fall","winter"])


# most of people prefer to rent bikes during fall season both during holidays and workingdays

# In[159]:


sns.kdeplot(df1.month,df1.totalcount,shade=True,cmap='Greens')


# In May,june and july most of the counts are between 4000 and 6000.
# In febreuary most of the counts are between 1950 and 2000.
# from august to september most of the counts are between 7500 and 8000.

# In[168]:


sns.barplot(data=df1[df1["holiday"]==1],x=df1.month,y=df1.totalcount)


# During Holiday season June and september have the highest number of counts.

# most of the people during holidays rent bikes between june and september

# In[79]:


ax=sns.barplot(x=df1.weathercondition,y=df1.totalcount)
ax.set_xticklabels(["clear","mist","light snow"])


# most of the people prefer to rent bikes when weather is clear or partly clouded.

# In[82]:


ax=sns.scatterplot(data=df1,x=df1["temp"],y=df1["totalcount"])


# warmer temperatures tend to have better chances of increasing counts

# In[83]:


ax=sns.scatterplot(data=df1,x=df1["windspeed"],y=df1["totalcount"])


# In[84]:


ax=sns.scatterplot(data=df1,x=df1["humidity"],y=df1["totalcount"])


# In[87]:


ax=sns.barplot(x=df1["season"],y=df1["temp"])
ax.set_xticklabels(["spring","summer","fall","winter"])


# In[97]:


ax=sns.barplot(x=df1["weathercondition"],y=df1["temp"])
ax.set_xticklabels(["clear","mist","light snow"])


# its clearly visible that season and temperature have a linear trend .temperatures increase at every season before dropping down in the winter season.

# feature selection

# In[436]:


from scipy.stats import chi2_contingency
from scipy.stats import f_oneway


# In[179]:


chi2,p,dof,ex= chi2_contingency(pd.crosstab(df1["holiday"],df1["workingday"]))
print(p)


# In[437]:


f_oneway(df1.totalcount,df1.weekday)


# the p value is less than 0.05. so it means that both the variables are dependent on each other. hence we will remove one of the two predictors for the model.

# In[92]:


df1=df1.drop(["date"],axis=1)


# In[ ]:





# date column is not required for buildig model.so we will remove it.

# MACHINE LEARNING

# In[397]:


from sklearn.model_selection import train_test_split
from sklearn import linear_model


# In[398]:


train,test = train_test_split(df1,test_size=0.2)


# In[399]:


train_attributes=train[['season','year',"month","weekday",'workingday','temp','windspeed','humidity','weathercondition','totalcount']]
test_attributes=test[['season','year','workingday',"month",'temp',"weekday",'windspeed','humidity','weathercondition','totalcount']]
cat_attributes=["year","season","workingday","weathercondition","month","weekday"]
num_attributes=["temp","windspeed","humidity","totalcount"]


# In[400]:


train_encoded_attributes=pd.get_dummies(train_attributes,columns=cat_attributes)
print("shape of data frame is" , train_encoded_attributes.shape )
train_encoded_attributes.columns


# In[401]:


y=pd.DataFrame(train_encoded_attributes["totalcount"])
train_encoded_attributes=train_encoded_attributes.drop(columns="totalcount")


# In[402]:


test_encoded_attributes=pd.get_dummies(test_attributes,columns=cat_attributes)
ytest=pd.DataFrame(test_encoded_attributes["totalcount"])
test_encoded_attributes=test_encoded_attributes.drop(columns="totalcount")


# In[403]:


train_encoded_attributes.shape


# In[404]:


lrmodel=linear_model.LinearRegression()
lrmodel


# In[405]:


lrmodel.fit(train_encoded_attributes.iloc[:,0:33],y.iloc[:,0])


# In[406]:


lrmodel.score(train_encoded_attributes,y)


# In[407]:


predictions_lr=lrmodel.predict(test_encoded_attributes.iloc[:,0:33])


# In[408]:


predictions_lr


# Error metrics
# 

# In[409]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(ytest.iloc[:,0].values,predictions_lr))
print("RMSE = " ,rmse)


# In[410]:


def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape


# In[411]:


print("mape = ", MAPE(ytest.iloc[:,0].values,predictions_lr))


# In[412]:


def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs((actual-predicted)))


# In[413]:


print("mae = ", mae(ytest.iloc[:,0].values,predictions_lr))


# In[414]:


residuals=[]
for i in range (0,len(ytest)):
    residuals.append(ytest.iloc[i]-predictions_lr[i])


# In[415]:


ig,ax=plt.subplots(figsize=(15,8))
ax.scatter(ytest,residuals)
ax.axhline(lw=2,color="black")
ax.set_xlabel("observed")
ax.set_ylabel("residuals")
ax.title.set_text("residual plot")
plt.show()


# Decision tree

# In[416]:


from sklearn.tree import DecisionTreeRegressor


# In[417]:


dt_model=DecisionTreeRegressor(max_depth=5).fit(train_encoded_attributes.iloc[:,0:33],y.iloc[:,0])


# In[418]:


dt_model.score(train_encoded_attributes,y)


# In[419]:


prediction_dt=dt_model.predict(test_encoded_attributes.iloc[:,0:33])


# In[420]:


prediction_dt


# In[421]:


print("MAPE = ",MAPE(ytest.iloc[:,0].values,prediction_dt))


# In[422]:


print("MAE = ",mae(ytest.iloc[:,0].values,prediction_dt))


# In[423]:


rmse=sqrt(mean_squared_error(ytest.iloc[:,0].values,prediction_dt))
print("RMSE = " ,rmse)


# In[424]:


residuals=[]
for i in range (0,len(ytest)):
    residuals.append(ytest.iloc[i]-prediction_dt[i])


# In[425]:


ig,ax=plt.subplots(figsize=(15,8))
ax.scatter(ytest,residuals)
ax.axhline(lw=2,color="black")
ax.set_xlabel("observed")
ax.set_ylabel("residuals")
ax.title.set_text("residual plot")
plt.show()


# Random forest

# In[426]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=200)


# In[427]:


rf.fit(train_encoded_attributes.iloc[:,0:33],y.iloc[:,0])


# In[428]:


predictions_rf=rf.predict(test_encoded_attributes.iloc[:,0:33])


# In[429]:


predictions_rf


# In[430]:


rf.score(train_encoded_attributes,y)


# In[431]:


print("MAPE = ",MAPE(ytest.iloc[:,0].values,predictions_rf))


# In[432]:


print("MAE = ",mae(ytest.iloc[:,0].values,predictions_rf))


# In[433]:


rmse=sqrt(mean_squared_error(ytest.iloc[:,0].values,predictions_rf))
print("RMSE = " ,rmse)


# In[434]:


residuals=[]
for i in range (0,len(ytest)):
    residuals.append(ytest.iloc[i]-predictions_rf[i])


# In[435]:


ig,ax=plt.subplots(figsize=(15,8))
ax.scatter(ytest,residuals)
ax.axhline(lw=2,color="black")
ax.set_xlabel("observed")
ax.set_ylabel("residuals")
ax.title.set_text("residual plot")
plt.show()


# From the Error metrics and the residual plots it can be observed that the model is performing best on the random forest model. hence the random forest model is best suited for this model.

# In[ ]:




