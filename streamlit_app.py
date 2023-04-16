import streamlit as st
import pandas as pd
import numpy as np
import joblib
# Importing all required modules 
import webbrowser as wb
import streamlit as st
import time
    
# -- Set page config
apptitle = 'Predict VN30-index price movement using financial news and technical analysis'
st.set_page_config(page_title=apptitle, 
                   page_icon="chart_with_upwards_trend",
                   layout="wide")

# Unpacking Scaler pkl file
S_file = open('model.pkl','rb')
scaler = joblib.load(S_file)

# Function to print out put which also converts numeric output from ML module to understandable STR 
def pred_out(num):
  if num == 1:
    st.info('THE VN30-INDEX WILL :green[BE UPTREND]', icon="ℹ️")
  elif num == 0:
    st.info('THE VN30-INDEX WILL BE :red[DOWNTREND]', icon="ℹ️")

st.title('Application :blue[Deep Learning] and :red[Machine Learning] in predicting VN30-index price movement using financial news and technical analysis')

###############################################################################################
import plotly.graph_objects as go
df = pd.read_csv('vn30-his-2.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

time_periods = {
    '5 years': pd.date_range(end=df.index[-1], periods=1260, freq=pd.tseries.offsets.BDay()),
    '1 year': pd.date_range(end=df.index[-1], periods=252, freq=pd.tseries.offsets.BDay()),
    '6 months': pd.date_range(end=df.index[-1], periods=120, freq=pd.tseries.offsets.BDay()),
    '3 month': pd.date_range(end=df.index[-1], periods=60, freq=pd.tseries.offsets.BDay()),
    '1 month': pd.date_range(end=df.index[-1], periods=20, freq=pd.tseries.offsets.BDay()),
    '2 week': pd.date_range(end=df.index[-1], periods=10, freq=pd.tseries.offsets.BDay()),
    '1 week': pd.date_range(end=df.index[-1], periods=5, freq=pd.tseries.offsets.BDay()),

}

fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
fig.update_layout(
    height=800,
    showlegend=True,
    title_text="VN30-Index Candlestick chart from 2017 to 2023",
)
########################################################################################
# Define dropdown menu label and options
dropdown_label = 'Select time period'
dropdown_options = list(time_periods.keys())

# Add dropdown menu to Streamlit app
time_period = st.selectbox(dropdown_label, dropdown_options)

# Filter data for selected time period
start_date = time_periods[time_period][0]
df_filtered = df.loc[start_date:]

# Update candlestick chart data
fig.update_traces(x=df_filtered.index,
                  open=df_filtered['Open'],
                  high=df_filtered['High'],
                  low=df_filtered['Low'],
                  close=df_filtered['Close'])
# ##########################################################################################
sma_10_trace = go.Scatter(x=df_filtered.index, y=df_filtered['sma_10'], name='SMA-10', visible=True)
sma_20_trace = go.Scatter(x=df_filtered.index, y=df_filtered['sma_20'], name='SMA-20', visible=True)
ema_10_trace = go.Scatter(x=df_filtered.index, y=df_filtered['ema_10'], name='EMA-10', visible=True)
ema_20_trace = go.Scatter(x=df_filtered.index, y=df_filtered['ema_20'], name='EMA-20', visible=True)

fig.add_trace(sma_10_trace)
fig.add_trace(sma_20_trace)
fig.add_trace(ema_10_trace)
fig.add_trace(ema_20_trace)
# Update figure layout to adjust legend and axis labels
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis_title="Date",
    yaxis_title="Price"
)

fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, theme="streamlit", use_container_width=True)
#########################################################################################
rsi_7_trace = go.Scatter(x=df_filtered.index, y=df_filtered['rsi_7'], name='RSI-7', visible=True)
rsi_9_trace = go.Scatter(x=df_filtered.index, y=df_filtered['rsi_9'], name='RSI-9', visible=True)
rsi_14_trace = go.Scatter(x=df_filtered.index, y=df_filtered['rsi_14'], name='RSI-14', visible=True)

fig_rsi = go.Figure(data=[rsi_7_trace,rsi_9_trace,rsi_14_trace])
# Update figure layout to adjust legend and axis labels
fig_rsi.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis_title="Date",
    yaxis_title="Price"
)

fig_rsi.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig_rsi, theme="streamlit", use_container_width=True)
#########################################################################################
fig2 = go.Figure(data=[go.Table(
    header=dict(values=list(df_filtered.columns)),
    cells=dict(values=[df_filtered.index, df_filtered.Close, df_filtered.Open,
                       df_filtered.High, df_filtered.Low, df_filtered.sma_10,
                       df_filtered.sma_20,df_filtered.ema_10,df_filtered.ema_20,
                       df_filtered.rsi_7, df_filtered.rsi_9, df_filtered.rsi_14]))
])

fig2.update_layout(
    height=400,
    showlegend=False,
    title_text="VN30-Index data table from 2017 to 2023",
)

# click_data = st.checkbox('Click here to show out all of historical data of VN30-Index')
# if click_data:
#   st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

st.sidebar.markdown('#### VN30-Index data table from 2017 to 2023')
click_data = st.sidebar.checkbox('Click here to show out all of historical data of VN30-Index', value=True)
if click_data:
  st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
##############################################################################################
st.header("Report model")
col00, col2, col3, col4, col5 = st.columns(5)
with col00:
  st.metric(label="", value="Label 0")
with col2:
  st.metric(label="Precison label 0", value="73%")
with col3:
  st.metric(label="Recall label 0", value="62%")
with col4:
  st.metric(label="F1-score", value="67%")
with col5:
  st.metric(label="Support", value="142")

col01, col6, col7, col8, col9 = st.columns(5)
with col01:
  st.metric(label="", value="Label 1")
with col6:
  st.metric(label="Precison label 1", value="70%")
with col7:
  st.metric(label="Recall label 1", value="80%")
with col8:
  st.metric(label="F1-score", value="75%")
with col9:
  st.metric(label="Support", value="161")

col13, col1, col10, col11, col12 = st.columns(5)
with col10:
  st.metric(label="Accuracy", value="71%")
with col11:
  st.metric(label="F1-score", value="71%")
with col12:
  st.metric(label="Support", value="303")  

import datetime

# d = st.date_input(
#     "When\'s your birthday",
#     datetime.date(2019, 7, 6))
# st.write('Your birthday is:', d)

st.header("Report model")
cold1, cold2 = st.columns(2)
with cold1:
  d = st.date_input(
    "Start: ",
    datetime.date(2019, 7, 6))
with cold2:
  d2 = st.date_input(
    "End: ",
    datetime.date(2023, 4, 4))

########################################################################
# option = st.selectbox(
#     '**How would you like to be input data?**',
#     ('Automatic','Manual Input', 'Upload a file'))

# if option == 'Manual Input':
# #   url = "https://finance.vietstock.vn/phan-tich-ky-thuat.htm#"
# #   if st.button('Click here to have exactly data of technical analysis ratio'):
# #     st.write(f"Finance.Vietstock analysis link: {url}")
#   with st.form("my_form"):
#     st.title("Historical data") 
#     # historical data
#     his1, his2, his3 = st.columns(3)
#     with his1:
#       bid_quality = st.number_input("Number of buy orders")
#     with his2:
#       bid_volume = st.number_input("Buy orders volume")
#     with his3:
#       ask_quality = st.number_input("Number of sell orders")
      
#     his4, his5, his6 = st.columns(3)
#     with his4:
#       ask_volume = st.number_input("Sell orders volume")
#     with his5:
#       matching_volume = st.number_input("Order matching volume")
#     with his6:
#       negotiable_volume = st.number_input("Put-through volume")
    
#     st.title("Financial news")
#     dum1, dum2, ta1 = st.columns(3)
#     with dum1:
#       positive = st.number_input("Positive news", value=1)
#     with dum2:
#       negative = st.number_input("Negative news", value=0)
#     with ta1:
#       SMA_10_lag = st.number_input("SMA 10 days")  
      
#     st.title("Technical analysis")
#     ta2, ta3, ta4 = st.columns(3)
#     with ta2:
#       SMA_20_lag = st.number_input("SMA 20 days")
#     with ta3:
#       EMA_10_lag = st.number_input("EMA 10 days")
#     with ta4:
#       EMA_20_lag = st.number_input("EMA 20 days")
      
#     ta5, ta6, ta7 = st.columns(3)
#     with ta5:
#       RSI_7d_lag = st.number_input("RSI 7 days")
#     with ta6:
#       RSI_9d_lag = st.number_input("RSI 9 days")
#     with ta7:
#       RSI_14d_lag = st.number_input("RSI 14 days")
#     # Prediction
#     features= [bid_quality, bid_volume, ask_quality, ask_volume, matching_volume, matching_volume,
#                positive, negative, SMA_10_lag, SMA_20_lag, EMA_10_lag, EMA_20_lag, RSI_7d_lag, RSI_9d_lag, RSI_14d_lag]
    
#     res_df = pd.DataFrame({'bid_quality':bid_quality, 'bid_volume':bid_volume, 'ask_quality':ask_quality, 'ask_volume':ask_volume,
#                            'matching_volume':matching_volume, 'negotiable_volume':negotiable_volume, 'Positive':positive, 'Negative':negative,
#                            'SMA_10':SMA_10_lag, 'SMA_20':SMA_20_lag, 'EMA_10':EMA_10_lag, 'EMA_20':EMA_20_lag, 'RSI_7d':RSI_7d_lag, 
#                            'RSI_9d':RSI_9d_lag, 'RSI_14d':RSI_14d_lag},index=["05-01-2023"])
    
#     pred = scaler.predict(np.array(features,ndmin=2))
    
#     submitted = st.form_submit_button("Forecast buttom") 
#     check_data = False
#     if submitted:     
#       if (RSI_14d_lag != 0.00 and bid_quality != 0.00 and bid_volume != 0.00 and ask_quality != 0.00 and ask_volume != 0.00 
#           and matching_volume != 0.00 and SMA_10_lag != 0.00 and SMA_20_lag != 0.00 and EMA_10_lag != 0.00 and EMA_20_lag != 0.00
#           and RSI_7d_lag != 0.00 and RSI_9d_lag != 0.00 and RSI_14d_lag != 0.00):
#         check_data = True
#         with st.spinner('Wait for it...'):
#           time.sleep(2)
#         st.success('This is a success updating!', icon="✅")
#         st.dataframe(res_df)
#         pred_out(pred)
#       else:
#         check_data = False
#         with st.spinner('Wait for it...'):
#           time.sleep(1)
#         st.warning('You do not input enough neccessary features', icon="⚠️")
    
# #     submitted_2 = st.form_submit_button("Get result") 
# #     if submitted_2:     
# #       with st.spinner('Wait for it...'):
# #         time.sleep(3)
# #       pred_out(pred)
# #     else:
# #       st.warning('You do not input enough neccessary features', icon="⚠️")

    
# elif option == 'Upload a file':
#   sample_df = pd.DataFrame({'Number of buy orders': 66774, 'Buy-orders volume':196533544, 'Number of sell orders':58645, 'Sell-orders volume':199406752,
#                            'Order matching volume':-2872869, 'Put-through volume':107108336, 'Positive':1, 'Negative':0,
#                            'SMA_10':1020, 'SMA_20':1019, 'EMA_10':1020, 'EMA_20':1019, 'RSI_7d':56, 
#                            'RSI_9d':55, 'RSI_14d':54},index=["dd-MM-YY"])
#   st.write("Please upload data like the sample:")
#   st.dataframe(sample_df)

#   uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
#   for uploaded_file in uploaded_files:
#     bytes_data = uploaded_file.read()
#     st.write("filename:", uploaded_file.name)
# else:
#     pass


select_event = st.sidebar.selectbox('Methods',
                                    ['Manual input', 'Upload file'])

if select_event == 'Manual input':
  with st.form("my_form"):
    st.title("Historical data") 
    # historical data
    his1, his2, his3 = st.columns(3)
    with his1:
      bid_quality = st.number_input("Number of buy orders")
    with his2:
      bid_volume = st.number_input("Buy orders volume")
    with his3:
      ask_quality = st.number_input("Number of sell orders")
      
    his4, his5, his6 = st.columns(3)
    with his4:
      ask_volume = st.number_input("Sell orders volume")
    with his5:
      matching_volume = st.number_input("Order matching volume")
    with his6:
      negotiable_volume = st.number_input("Put-through volume")
    
    st.title("Financial news")
    dum1, dum2, ta1 = st.columns(3)
    with dum1:
      positive = st.number_input("Positive news", value=1)
    with dum2:
      negative = st.number_input("Negative news", value=0)
    with ta1:
      SMA_10_lag = st.number_input("SMA 10 days")  
      
    st.title("Technical analysis")
    ta2, ta3, ta4 = st.columns(3)
    with ta2:
      SMA_20_lag = st.number_input("SMA 20 days")
    with ta3:
      EMA_10_lag = st.number_input("EMA 10 days")
    with ta4:
      EMA_20_lag = st.number_input("EMA 20 days")
      
    ta5, ta6, ta7 = st.columns(3)
    with ta5:
      RSI_7d_lag = st.number_input("RSI 7 days")
    with ta6:
      RSI_9d_lag = st.number_input("RSI 9 days")
    with ta7:
      RSI_14d_lag = st.number_input("RSI 14 days")
    # Prediction
    features= [bid_quality, bid_volume, ask_quality, ask_volume, matching_volume, matching_volume,
               positive, negative, SMA_10_lag, SMA_20_lag, EMA_10_lag, EMA_20_lag, RSI_7d_lag, RSI_9d_lag, RSI_14d_lag]
    
    res_df = pd.DataFrame({'bid_quality':bid_quality, 'bid_volume':bid_volume, 'ask_quality':ask_quality, 'ask_volume':ask_volume,
                           'matching_volume':matching_volume, 'negotiable_volume':negotiable_volume, 'Positive':positive, 'Negative':negative,
                           'SMA_10':SMA_10_lag, 'SMA_20':SMA_20_lag, 'EMA_10':EMA_10_lag, 'EMA_20':EMA_20_lag, 'RSI_7d':RSI_7d_lag, 
                           'RSI_9d':RSI_9d_lag, 'RSI_14d':RSI_14d_lag},index=["05-01-2023"])
    
    pred = scaler.predict(np.array(features,ndmin=2))
    
    submitted = st.form_submit_button("Forecast buttom") 
    check_data = False
    if submitted:     
      if (RSI_14d_lag != 0.00 and bid_quality != 0.00 and bid_volume != 0.00 and ask_quality != 0.00 and ask_volume != 0.00 
          and matching_volume != 0.00 and SMA_10_lag != 0.00 and SMA_20_lag != 0.00 and EMA_10_lag != 0.00 and EMA_20_lag != 0.00
          and RSI_7d_lag != 0.00 and RSI_9d_lag != 0.00 and RSI_14d_lag != 0.00):
        check_data = True
        with st.spinner('Wait for it...'):
          time.sleep(2)
        st.success('This is a success updating!', icon="✅")
        st.dataframe(res_df)
        pred_out(pred)
      else:
        check_data = False
        with st.spinner('Wait for it...'):
          time.sleep(1)
        st.warning('You do not input enough neccessary features', icon="⚠️")
else:
  sample_df = pd.DataFrame({'Number of buy orders': 66774, 'Buy-orders volume':196533544, 'Number of sell orders':58645, 'Sell-orders volume':199406752,
                           'Order matching volume':-2872869, 'Put-through volume':107108336, 'Positive':1, 'Negative':0,
                           'SMA_10':1020, 'SMA_20':1019, 'EMA_10':1020, 'EMA_20':1019, 'RSI_7d':56, 
                           'RSI_9d':55, 'RSI_14d':54},index=["dd-MM-YY"])
  st.write("Please upload data like the sample:")
  st.dataframe(sample_df)

  uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
  for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)


