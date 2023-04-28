import streamlit as st
import pandas as pd
import numpy as np
import joblib
# Importing all required modules 
import webbrowser as wb
import requests
from bs4 import BeautifulSoup

# -- Set page config
apptitle = 'Predict VN30-index crawl data'
st.set_page_config(page_title=apptitle, 
                   layout="wide",
                   page_icon="🧊",
                   initial_sidebar_state="expanded")
#                  page_icon="chart_with_upwards_trend")

# Crawl the webpage using requests
url = 'https://vnexpress.net/thanh-tich-kinh-te-khong-tron-ven-cua-ong-biden-4598764.html'
response = requests.get(url)
# Parse the HTML using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Extract the data you want to display
title = soup.title.string
# Display the data using Streamlit
st.title(title)
