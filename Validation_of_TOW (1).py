
# coding: utf-8

# In[1]:


"""
necessary libraries
"""

import seaborn as sns
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("training.csv",sep='\t',parse_dates=True,index_col="DepartureDate") # this is to create our model


# In[197]:


data1 = pd.read_csv("validation.csv",sep="\t",parse_dates=True,index_col="DepartureDate") #the data to be validated


# In[4]:


"""
number of the columns 
"""

len(data.columns.tolist())


# In[5]:


"""
this part changes the (null) values to NaN with numpy's NaN so the missing data can be handled

in our case we're only interested in last  columns 

"""
for i in range(-5,0):
    data.iloc[:,i] = data.iloc[:,i].replace("(null)",np.NaN)


# In[6]:


data.info()


# In[7]:


data.head()


# In[8]:


"""
here I get only the interested columns by using pandas concat method.

"""

df = pd.concat([data["ActualTotalFuel"],data["ActualTOW"],data["FLownPassengers"],data["BagsCount"],data["FlightBagsWeight"]],axis=1)


# In[9]:


df.head()


# In[10]:


df = df.apply(pd.to_numeric) # convert the entire dataframe into numeric datatype


# In[11]:


df.info()


# In[195]:


"""
next I take the mean of the each row so that I can replace the NaN values with the mean of each.
To get the mean  I used np.nanmean() which takes the mean without taking into account the NaN values


"""


# In[12]:


mean_flightbag = np.nanmean(df["FlightBagsWeight"])


# In[13]:


mean_flightbag


# In[14]:


mean_bags = np.nanmean(df["BagsCount"])


# In[15]:


mean_bags


# In[16]:


mean_flown = np.nanmean(df["FLownPassengers"])


# In[17]:


mean_flown 


# In[18]:


mean_actualTow = np.nanmean(df["ActualTOW"])


# In[19]:


mean_actualTow


# In[20]:


df["FlightBagsWeight"].fillna(mean_flightbag,inplace=True)
df["BagsCount"].fillna(mean_bags,inplace=True)
df["FLownPassengers"].fillna(mean_flown,inplace=True)
df["ActualTOW"].fillna(mean_actualTow,inplace=True)


# In[21]:


df.info()


# In[22]:


df.head()


# In[207]:


sn = sns.pairplot(df,size=5)

sn.savefig("Data.jpeg")


# In[23]:


sns.heatmap(df.corr(),linewidths=0.3, vmax = 1.0,square=True,linecolor="red",annot=True)


# In[31]:


"""
in the next step I take the features and label

ActualTOW is the label and the rest is the feature
"""

features = df.iloc[:,:-1].values

label = df.iloc[:,-1].values


# In[32]:


features.shape


# In[33]:


features


# In[34]:


label.shape


# In[35]:


label


# In[36]:


df.head() # I have changed the order of the columns inorder to visualzie dependent variable  better


# In[229]:


"""
cross validation part
"""


# In[69]:


from sklearn.cross_validation import train_test_split


# In[92]:


X_train, X_test, y_train, y_test = train_test_split(features,label,test_size=0.3,random_state=0)


# In[39]:


from sklearn.linear_model import LinearRegression


# In[40]:


regressor = LinearRegression()


# In[41]:


regressor.fit(X_train,y_train)


# In[42]:


y_pred = regressor.predict(X_test)


# In[43]:


y_pred # here we have the predicted values 


# In[44]:


label # here we have the actual values 


# In[45]:


coeff = regressor.coef_
intercpt = regressor.intercept_


# In[46]:


coeff


# In[47]:


intercpt


# In[51]:


train_score = regressor.score(X_train,y_train)
test_score  = regressor.score(X_test,y_test)


# In[52]:


test_score


# In[53]:


train_score


# In[54]:


from sklearn import metrics


# In[55]:


mse = metrics.mean_squared_error(y_test,y_pred)


# In[56]:


mse


# In[57]:


rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))


# In[58]:


rmse


# In[59]:


len(regressor.coef_)


# In[70]:


regressor.coef_


# In[112]:


df.head()


# In[115]:


plt.scatter(y_test,y_pred) # to visualize the predictions


# In[192]:


"""
Here I take 18 samples from dataframe and make predictions, firstly I converted them into 2d numpy arrays 
then computed the prediction
"""
from copy import deepcopy
gh = deepcopy(df.head(18))
lk = gh.as_matrix(columns=["FLownPassengers","BagsCount","FlightBagsWeight","ActualTotalFuel"]) # here I convert the df to 2d array

hold = regressor.predict(lk) # making predictions
data_c = pd.DataFrame(hold) # creating dataframe
hold1 = pd.DataFrame(df.iloc[:18,4].values) # getting the first 18 ActualTOW 
concated = pd.concat((data_c,hold1),axis=1,ignore_index=True) # here I creaet a dataframe by concatinating two data frames
concated.columns = ["Predicted","ActualTOW"]  # renaming the columns 


# In[193]:


"""
since we only have 18 predictions scatter wouldn't tell us much so by looking at the line graph we can observe how close the
lines to eachother
"""
concated.plot()
plt.show()


# In[187]:


concated # here we can see the predictions and ActualTOW dataframe 


# In[201]:


"""
first I read the data and determine the columns that I will be working with next, I will be making data cleaning in order to get
more precise results replacing the (null) with NaN and then NaN with the mean of the each column 
"""
data1.head(2)


# In[199]:


"""
this part replaces the (null) with NaN
"""
for i in range(-5,0):
    data1.iloc[:,i] = data1.iloc[:,i].replace("(null)",np.NaN)


# In[200]:


data1.info()


# In[202]:


"""
here I get only the interested columns by using pandas concat method.

"""
df1 = pd.concat([data1["ActualTotalFuel"],data1["FLownPassengers"],data1["BagsCount"],data1["FlightBagsWeight"]],axis=1)
df1 = df1.apply(pd.to_numeric) # convert the entire dataframe into numeric datatype


# In[203]:


df1.info()


# In[205]:


mean_flightbag1 = np.nanmean(df1["FlightBagsWeight"])
mean_bags1 = np.nanmean(df1["BagsCount"])
mean_flown1 = np.nanmean(df1["FLownPassengers"])


# In[206]:


print("mean flightbag: {} mean bags: {} mean flownpassengers: {}".format(mean_flightbag1,mean_bags1,mean_flown1))


# In[207]:


df1["FlightBagsWeight"].fillna(mean_flightbag1,inplace=True)
df1["BagsCount"].fillna(mean_bags1,inplace=True)
df1["FLownPassengers"].fillna(mean_flown1,inplace=True)


# In[208]:


df1.info()


# In[212]:


"""
in the next step I convert the dataframe into 2d numpy array to predict the ActualTOW 
"""
data_to_predict = df1.as_matrix(columns=["FLownPassengers","BagsCount","FlightBagsWeight","ActualTotalFuel"])
data_to_predict


# In[214]:


predicted = regressor.predict(data_to_predict) # here I use the trained model to predict the ActualTOW


# In[216]:


df_predict = pd.DataFrame(predicted) # here I create a dataframe which holds the predicted values


# In[217]:


df_predict.columns= ["PredictedTOW"] # renaming the columnn of it


# In[219]:


df_predict_values = df_predict.values # getting the values to add a column to our actual dataframe df1


# In[221]:


df1["PredictedTOW"] = df_predict_values # adding the column


# In[228]:


df1.head()


# In[225]:


df1.info()


# In[ ]:


df1.to_csv("validation_calculated.csv")

