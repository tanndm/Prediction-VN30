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
    st.warning('THE VN30-INDEX WILL :green[BE UPTREND]')
  else:
    st.success('THE VN30-INDEX WILL BE :red[DOWNTREND]')

st.title('Application :blue[Deep Learning] and :red[Machine Learning] in predicting VN30-index price movement using financial news and technical analysis')



chart_data = pd.read_csv("https://raw.githubusercontent.com/BrianNguyen2001/Crawling-data-from-web/main/historical-vn30.csv",index_col=0)
st.line_chart(chart_data)

st.header("Reoprt model")

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

with col11:
  st.metric(label="Accuracy", value="71%")
with col12:
  st.metric(label="Support", value="303")  
  

option = st.selectbox(
    '**How would you like to be input data?**',
    ('Manual Input', 'Upload a file'))

if option == 'Manual Input':
  url = "https://finance.vietstock.vn/phan-tich-ky-thuat.htm#"
  if st.button('Click here to have exactly data of technical analysis ratio'):
    st.write(f"Finance.Vietstock analysis link: {url}")

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
#     features= ['bid_quality', 'bid_volume', 'ask_quality', 'ask_volume', 'matching_volume', 'matching_volume',
#                'Positive', 'Negative', 'SMA_10_lag', 'SMA_20_lag', 'EMA_10_lag', 'EMA_20_lag', 'RSI_7d_lag', 'RSI_9d_lag', 'RSI_14d_lag']
    
    res_df = pd.DataFrame({'bid_quality':bid_quality, 'bid_volume':bid_volume, 'ask_quality':ask_quality, 'ask_volume':ask_volume,
                           'matching_volume':matching_volume, 'negotiable_volume':negotiable_volume, 'Positive':positive, 'Negative':negative,
                           'SMA_10':SMA_10_lag, 'SMA_20':SMA_20_lag, 'EMA_10':EMA_10_lag, 'EMA_20':EMA_20_lag, 'RSI_7d':RSI_7d_lag, 
                           'RSI_9d':RSI_9d_lag, 'RSI_14d':RSI_14d_lag},index=["05-01-2023"])
#     pred = scaler.predict(np.array(features,ndmin=2))
    
    submitted = st.form_submit_button("Submit data") 
    if submitted:     
#       with st.spinner('Wait for it...'):
#         time.sleep(5)
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
      st.info('Next day, VN30-index will uptrend', icon="ℹ️")
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




