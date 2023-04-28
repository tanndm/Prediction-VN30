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
# Get the URL from the user using a text input widget
url = st.text_input('Enter a URL')

# Extract the href attribute from the link using requests and BeautifulSoup
if url:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a')
    hrefs = [link.get('href') for link in links]

    # Display the extracted href attributes using a write widget
    st.write('Extracted href attributes:')
    link_list = set(list(hrefs))
    else_list = []
    for href in link_list:
      if ("https://" in href) and ('-' in href):
        else_list.append(href)
    st.write(else_list)
    
    for i in else_list:
      response_2 = requests.get(i)
      soup_2 = BeautifulSoup(response_2.content, 'html.parser')

    # Extract the data you want to display
      title_list = []
      title = soup_2.title.string
      title_list.append(title)
      st.write(title)
    # Display the data using Streamlit
#     st.title(title_list)
