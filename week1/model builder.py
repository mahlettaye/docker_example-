#importing module
import helper as hl
from helper import Helper as h
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump,load
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


data=hl.read_csv('data/winequality-red.csv') 

data=h.replce_with_median(data)


# Python file creates dummy data for train and test 

# no of categorical columns
cat = data.select_dtypes(include='O')
# create dummies of categorical columns
df_dummies = pd.get_dummies(data,drop_first = True)



df_dummies['best quality']=[1 if x>=7 else 0 for x in data.quality]
# independent variables
x = df_dummies.drop(['quality','best quality'],axis=1)
# dependent variable
y = df_dummies['best quality']
  
# creating train test splits
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=40)



# creating scaler scale var.
norm = MinMaxScaler()
# fit the scal
norm_fit = norm.fit(xtrain)
dump(norm_fit,'model/scaler.joblib')
# transfromation of trainig data
scal_xtrain = norm_fit.transform(xtrain)

# transformation of testing data
scal_xtest = norm_fit.transform(xtest)
print(scal_xtrain)




  
# create model variable
rnd = RandomForestClassifier()
  
# fit the model
fit_rnd = rnd.fit(scal_xtrain,ytrain)
  
# checking the accuracy score
rnd_score = rnd.score(scal_xtest,ytest) 
  
print('score of model is : ',rnd_score)
  
dump(rnd, 'model/model.joblib')
x_predict = list(rnd.predict(xtest))
df = {'predicted':x_predict,'orignal':ytest}

