import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import time
from stqdm import stqdm
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json

st.set_page_config(layout="wide",page_title="Omdena Berlin", page_icon="flag.png", initial_sidebar_state="expanded")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

css_style = {
    "icon": {"color": "white"},
    "nav-link": {"--hover-color": "grey"},
    "nav-link-selected": {"background-color": "#FF4C1B"},
}

img_germany = Image.open("logo.jpg")




cfg_model_path = 'models/best.pt'
model = None
confidence = .45


def image_input():
    img_file = None
    img_bytes = st.file_uploader("Upload an image", type=['png','jpeg','jpg'])
    if img_bytes:
        img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
        Image.open(img_bytes).save(img_file)
    if img_file:
        col1,col2 = st.columns(2)
        with col1:
            st.image(img_file, caption = "Raw Image")
        with col2:
            img = infer_image(img_file)
            st.image(img, caption="Severity Prediction")
           

def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    
    return image


def load_model(path):

    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_.to(device)

    return model_


def contributors_page():
    st.write("""
                <h1 style="text-align: center; color:#008000;">A heartfelt thankyou to all our contributors ❤️</h1><hr>
                <div style="text-align:center;">
                <table>
                <tr>
                    <th width="20%" style="font-size: 140%;">Chapter Name</th>    
                    <th width="20%" style="font-size: 140%;">Chapter Lead</th>    
                </tr>
                <tr>
                    <td>Berlin, Germany Local Chapter</td>    
                    <td>Vishu kalier</td>    
                </tr>
                </table>
                <br>
                <table>
                    <tbody>
                        <tr>
                            <th width="20%" style="font-size: 140%;">Task Name</th>
                            <th width="20%" style="font-size: 140%;">Task Lead</th>
                        </tr>
                        <tr>
                            <td>Knowledge</td>
                            <td>Vishu kalier</td>
                        </tr>
                        <tr>
                            <td>Data Collection</td>
                            <td>Pritam Bhakta, Bibhuti Baibhav Borah</td>
                        </tr>
                        <tr>
                            <td>Data Preprocessing</td>
                            <td>Pritam Bhakta</td>
                        </tr>
                        <tr>
                            <td>Data Analysis</td>
                            <td>Pritam Bhakta</td>
                        </tr>
                        <tr>
                            <td>Deploying</td>
                            <td>Bibhuti Baibhav Borah</td>
                        </tr>
                    </tbody>
                </table>
                </div>
                <hr>
            """, unsafe_allow_html=True)

def about_page():
    st.image(img_germany)
    st.write("""<h1>Project background</h1>""", unsafe_allow_html=True)
    st.write("""
        <p>Burns are among the most prevalent skin issues encountered
              in daily life, with a variety of causes such as boiling
              water, electricity, and UV rays. The severity of burns 
             depends on both the victim and the source of the injury. 
             Burns are classified based on their severity, ranging from 
             mild cases that may present as a simple rash or boil to severe
              cases resulting from electric shocks that can lead to the breakdown
              of the skin’s epidermis and capillaries, potentially proving fatal.
              While first aid is often used to treat burns, more severe cases necessitate
              immediate medical attention.</p><br>
    """, unsafe_allow_html=True)
    st.write(  """
             <p>The goal is to create a Model that accurately detects the source and 
             severity of the burn.- Build a website or application which can
              be used by any person or victim.- Making the collaborators learn
              new skills and advance into data science and Artificial Intelligence.
             Making inexperienced people get hands-on with Neural Networks, Image Recognition, 
             and the Medical field as well.</p><br>
    """, unsafe_allow_html=True)

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def Home_page():
    global model, confidence, cfg_model_path
    
    st.title("Skin Burn Severity detection 🔎")

    lottie_ani = load_lottiefile("animation1.json")
    st_lottie(
        lottie_ani,
        speed =1,
        reverse=False,
        quality= "high",
        height= 160,
        width=160

    )
    
    

    # device options
    
    model = load_model(cfg_model_path)

    

    image_input()


with st.sidebar:
    st.sidebar.title("Settings🛠️")
    selected = option_menu(
    menu_title=None,
    options=["Home","About", "Contributors"],
    icons=["house","info-circle", "people"],
    styles=css_style
    )
        
if selected == "Home":
    Home_page()

elif selected == "About":
    
    about_page()

elif selected == "Contributors":
    contributors_page()    
































































































