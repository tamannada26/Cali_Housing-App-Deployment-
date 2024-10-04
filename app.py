import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

#load the dataset
cal=fetch_california_housing()
df=pd.DataFrame(cal.data,columns=cal.feature_names)

df['price']=cal.target
df.head()

#whenever we use st. something, it will display on the frontend
#title of the app
st.title('California House price prediction for XYZ brokerage company')

#data overview

st.subheader("Data Overview")
st.dataframe(df.head(10))
#split the data into train and test
# split the data into two parts input & output

X=df.drop(['price'], axis=1) # input cols

y=df['price'] # output cols
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#standardize the data
scaler=StandardScaler()

X_train_sc=scaler.fit_transform(X_train)
X_test_sc=scaler.transform(X_test)

#model selection
st.subheader("## SELECT A MODEL")

model=st.selectbox("Choose a model",["Linear Regression","Ridge","Lasso","Elastic Net"])

#initialize  the model
models={
    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Elastic Net":ElasticNet()
}
# train the selected model
selected_model=models[model]

# train the model
selected_model.fit(X_train_sc,y_train)


#predict the values
y_pred=selected_model.predict(X_test_sc)

#evaluating metrics

test_r2=print("r2_score", r2_score(y_test, y_pred))
test_mse=print("mean_squared_error", mean_squared_error(y_test, y_pred))
test_mae=print("mean_absolute_error", mean_absolute_error(y_test, y_pred))
test_rmse=print("RMSE", np.sqrt(mean_squared_error(y_test, y_pred)))

#display the metrics for selected model
st.write("Test MSE:",test_mse)
st.write("Test MAE:",test_mae)
st.write("Test RMSE:",test_rmse)
st.write("Test R2",test_r2)

# prompt the user for input
st.write("Enter the input values to predict the house price")


user_input={}
for features in X.columns:
    user_input[features]= st.number_input(features)

# convert the dictiuonary to dataframe bcz our trained datai.e. X_train,X_test,etc are in dataframe
user_input_df=pd.DataFrame([user_input])

#scale the user input
user_input_sc= scaler.transform(user_input_df)
# predict the house price
predicted_price=selected_model.predict(user_input_sc)
#displaying the predicted price
st.write(f"Predicted house price is:{predicted_price[0]*100000}")