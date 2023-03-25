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
    '1 week': pd.date_range(end=df.index[-1], periods=5, freq=pd.tseries.offsets.BDay()),
    '1 month': pd.date_range(end=df.index[-1], periods=20, freq=pd.tseries.offsets.BDay()),
    '6 months': pd.date_range(end=df.index[-1], periods=120, freq=pd.tseries.offsets.BDay()),
    '1 year': pd.date_range(end=df.index[-1], periods=252, freq=pd.tseries.offsets.BDay()),
    '5 years': pd.date_range(end=df.index[-1], periods=1260, freq=pd.tseries.offsets.BDay())
}

fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

fig.update_layout(
    height=800,
    showlegend=False,
    title_text="VN30-Index Candlestick chart from 2017 to 2023",
)

########################################################################################
# Define dropdown menu label and options
dropdown_label = 'Select time period'
dropdown_options = list(time_periods.keys())

# Add dropdown menu to Streamlit app
time_period = st.selectbox(dropdown_label, dropdown_options)

st.write(time_periods[time_period][0])
# Filter data for selected time period
start_date = time_periods[time_period][0]
df_filtered = df.loc[start_date:]

# # Update candlestick chart data
# fig.update_traces(x=df_filtered.index,
#                   open=df_filtered['Open'],
#                   high=df_filtered['High'],
#                   low=df_filtered['Low'],
#                   close=df_filtered['Close'])

# ##########################################################################################
# sma_10_trace = go.Scatter(x=df['Date'], y=df['sma_10'], name='SMA-10', visible=True)
# sma_20_trace = go.Scatter(x=df['Date'], y=df['sma_20'], name='SMA-20', visible=True)
# ema_10_trace = go.Scatter(x=df['Date'], y=df['ema_10'], name='EMA-10', visible=True)
# ema_20_trace = go.Scatter(x=df['Date'], y=df['ema_20'], name='EMA-20', visible=True)
# rsi_7_trace = go.Scatter(x=df['Date'], y=df['rsi_7'], name='RSI-7', visible=True)
# rsi_9_trace = go.Scatter(x=df['Date'], y=df['rsi_9'], name='RSI-9', visible=True)
# rsi_14_trace = go.Scatter(x=df['Date'], y=df['rsi_14'], name='RSI-14', visible=True)

# fig.add_trace(sma_10_trace)
# fig.add_trace(sma_20_trace)
# fig.add_trace(ema_10_trace)
# fig.add_trace(ema_20_trace)
# fig.add_trace(rsi_7_trace)
# fig.add_trace(rsi_9_trace)
# fig.add_trace(rsi_14_trace)

# # Define button label and default visibility
# button_label = 'Show all Technical analysis indicators'
# visible = True

# # Add button to Streamlit app
# if st.button(button_label):
#     visible = not visible
#     if visible:
#         sma_10_trace.visible = True
#         sma_20_trace.visible = True
#         ema_10_trace.visible = True
#         ema_20_trace.visible = True
#         rsi_7_trace.visible = True
#         rsi_9_trace.visible = True
#         rsi_14_trace.visible = True
#         button_label = 'Show Technical analysis indicators'
#     # Update figure layout to adjust trace visibility
#     fig.update_layout(showlegend=True)

# # Update figure layout to adjust legend and axis labels
# fig.update_layout(
#     legend=dict(
#         orientation="h",
#         yanchor="bottom",
#         y=1.02,
#         xanchor="right",
#         x=1
#     ),
#     xaxis_title="Date",
#     yaxis_title="Price"
# )

fig.update_layout(xaxis_rangeslider_visible=False)

st.plotly_chart(fig, theme="streamlit", use_container_width=True)
#########################################################################################
df_2 = pd.read_csv('vn30-his-2.csv')
fig2 = go.Figure(data=[go.Table(header=dict(
              values=["Date","Close", "Open",
                    "High", "Low",'sma_10', 
                    'sma_20', 'ema_10', 'ema_20',
                     'rsi_7', 'rsi_9', 'rsi_14']),
              cells=dict(values=[df_2[k].tolist() for k in df_2.columns[:-1]]))
                     ])

    
fig2.update_layout(
    height=400,
    showlegend=False,
    title_text="VN30-Index data table from 2017 to 2023",
)

click_data = st.checkbox('Click here to show out all of historical data of VN30-Index')
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
  
########################################################################
option = st.selectbox(
    '**How would you like to be input data?**',
    ('Manual Input', 'Upload a file'))

if option == 'Manual Input':
#   url = "https://finance.vietstock.vn/phan-tich-ky-thuat.htm#"
#   if st.button('Click here to have exactly data of technical analysis ratio'):
#     st.write(f"Finance.Vietstock analysis link: {url}")
  with st.form("my_form"):
    st.title("Historical data") 
    # historical data
    his1, his2, his3 = st.columns(3)
    with his1:
      bid_quality = st.number_input("Feature 1")
    with his2:
      bid_volume = st.number_input("Feature 2")
    with his3:
      ask_quality = st.number_input("Feature 3")
      
    his4, his5, his6 = st.columns(3)
    with his4:
      ask_volume = st.number_input("Feature 4")
    with his5:
      matching_volume = st.number_input("Feature 5")
    with his6:
      negotiable_volume = st.number_input("Feature 6")
    
    st.title("Financial news")
    dum1, dum2, ta1 = st.columns(3)
    with dum1:
      positive = st.number_input("Feature 7", value=1)
    with dum2:
      negative = st.number_input("Feature 8", value=0)
    with ta1:
      SMA_10_lag = st.number_input("Feature 9")  
      
    st.title("Technical analysis")
    ta2, ta3, ta4 = st.columns(3)
    with ta2:
      SMA_20_lag = st.number_input("Feature 10")
    with ta3:
      EMA_10_lag = st.number_input("Feature 11")
    with ta4:
      EMA_20_lag = st.number_input("Feature 12")
      
    ta5, ta6, ta7 = st.columns(3)
    with ta5:
      RSI_7d_lag = st.number_input("Feature 13")
    with ta6:
      RSI_9d_lag = st.number_input("Feature 14")
    with ta7:
      RSI_14d_lag = st.number_input("Feature 15")
    # Prediction
    features= [bid_quality, bid_volume, ask_quality, ask_volume, matching_volume, matching_volume,
               positive, negative, SMA_10_lag, SMA_20_lag, EMA_10_lag, EMA_20_lag, RSI_7d_lag, RSI_9d_lag, RSI_14d_lag]
    
    res_df = pd.DataFrame({'bid_quality':bid_quality, 'bid_volume':bid_volume, 'ask_quality':ask_quality, 'ask_volume':ask_volume,
                           'matching_volume':matching_volume, 'negotiable_volume':negotiable_volume, 'Positive':positive, 'Negative':negative,
                           'SMA_10':SMA_10_lag, 'SMA_20':SMA_20_lag, 'EMA_10':EMA_10_lag, 'EMA_20':EMA_20_lag, 'RSI_7d':RSI_7d_lag, 
                           'RSI_9d':RSI_9d_lag, 'RSI_14d':RSI_14d_lag},index=["05-01-2023"])
    
    pred = scaler.predict(np.array(features,ndmin=2))
    
    submitted = st.form_submit_button("Submit data") 
    if submitted:     
      if (RSI_14d_lag != 0.00):
        with st.spinner('Wait for it...'):
          time.sleep(3)
        st.success('This is a success updating!', icon="✅")
        st.dataframe(res_df)
      else:
        with st.spinner('Wait for it...'):
          time.sleep(1)
        st.warning('You do not input enough neccessary features', icon="⚠️")
    
    submitted_2 = st.form_submit_button("Get result") 
    if submitted_2:     
      with st.spinner('Wait for it...'):
        time.sleep(3)
      pred_out(pred)
#       st.info(pred)

else:
  uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
  for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
  
st.title("Train model")


        


col1_1, col2_1, = st.columns(2)
with col1_1:
  st.metric(label="VN30-Index", value= "14-04-2023")
with col2_1:
  st.metric(label="Trend Prediction", value="80%")




