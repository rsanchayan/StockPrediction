import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as datareader
from datetime import date, timedelta
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


st.title("Stock Trend Prediction")
inp = st.text_input('Enter Stock Ticker','AAPL')
start = '2010-01-01'
end=date.today()
data=datareader.DataReader('AAPL','stooq',start,end)
data1=datareader.DataReader(inp,'stooq',start,end)

data1.set_axis(['Date', 'Open', 'High','Low','Close','Volume'], axis='columns', inplace=True)

# data.sort_values(by=['Date'],ascending=False)
# data1.sort_values(by=data1.iloc[:,0])

if st.button("Click Here to View Stock Details"):
    st.subheader('Stock Details from 2010')
    st.write(data1.info())


data=data.reset_index()
data1=data1.reset_index()
data['Date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')
data1['Date'] = pd.to_datetime(data1['Date'], format = '%Y%m%d')
data['day'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year

data1['day'] = data1['Date'].dt.day
data1['month'] = data1['Date'].dt.month
data1['year'] = data1['Date'].dt.year
data['ma100'] = data.Close.rolling(100).mean()
data['ma200'] = data.Close.rolling(200).mean()
data1['ma200'] = data1.Close.rolling(200).mean()
data1['ma100'] = data1.Close.rolling(100).mean()
data1=data1.dropna()
data=data.dropna()
Index=data.Date





st.subheader("Closing Price Trend")
fig1 = plt.figure(figsize=(12,6))
plt.plot(data1.Close)
st.pyplot(fig1)

st.subheader("Closing Price Trend Along With 100 Days Moving Average ")
fig2=plt.figure(figsize=(12, 6))
plt.plot(data1.Close)
plt.plot(data1['ma100'])
st.pyplot(fig2)

st.subheader("Closing Price Trend Along With 100 & 200 Days Moving Average ")
fig3=plt.figure(figsize=(12, 6))
plt.plot(data1.Close)
plt.plot(data1['ma100'])
plt.plot(data1['ma200'])
st.pyplot(fig3)

#ML DATA PRE-PROCESSING
X=data[['day','month','year','ma100','ma200']]
y=data['Close']
X=X.dropna()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
Model = LinearRegression().fit(X_train,y_train)
y.index=Index
data1=data1.dropna()
Index1=data1.Date
output = pd.DataFrame(Model.predict(data1[['day','month','year','ma100','ma200']]))
output.index=Index1
data1.index=Index1

st.subheader("Closing Price Predicted Trend")
fig4=plt.figure(figsize=(12, 6))
plt.plot(output, 'r' ,label= 'Predicted Price')
plt.plot(data1.Close, 'b' ,label= 'Original Price')
st.pyplot(fig4)

st.subheader("Check Tomorrow's Predicted Share Price")
day=date.today()+timedelta(1)
bbb=data1.iloc[-1]
aaa=date.today()-timedelta(1)
string_date=aaa.strftime('%Y-%m-%d')
tomorrow_prediction=(Model.predict([[day.day,day.month,day.year,bbb['ma100'],bbb['ma200']]]))
price="{:.2f}".format(tomorrow_prediction[0])
if st.button("Click Here to Check Tomorrow's Predicted Share Price"):
    st.write('Predicted Share price for tomorrow is',price )
