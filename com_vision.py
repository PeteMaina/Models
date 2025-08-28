'''This code runs only on google colab
the code uses free intelligent API's to generate actual faces with real backgrounds
This gives the Ilussion that the person is real
The free API's have a limit to only generate 1000 image requests per day
The code also uses free API's to generate random names and bios for the generated faces
there is also a button to get a new face with a new name and bio
This is just an illustration of how powerful AI is
the code can be modified to use paid API's to generate more realistic faces and more details
it gets the job done for free and thats why it has 2500 lines of code'''

!pip install requests
!pip install Pillow
!pip install google.colab
!pip install matplotlib
!pip install numpy
!pip install opencv-python
!pip install deepface
!pip install fer
!pip install mtcnn
!pip install tensorflow
!pip install keras
!pip install dlib
!pip install face_recognition
!pip install imutils
!pip install mediapipe
!pip install imageio
!pip install imageio[ffmpeg]
!pip install moviepy
!pip install gTTS   
!pip install playsound
!pip install pyttsx3
!pip install pywhatkit
!pip install SpeechRecognition
!pip install pydub
!pip install transformers
!pip install diffusers
!pip install accelerate
!pip install safetensors
!pip install scipy
!pip install ftfy
!pip install gradio
!pip install openai
!pip install python-dotenv
!pip install langchain
!pip install chromadb
!pip install PyPDF2
!pip install pdfplumber
!pip install llama-index
!pip install beautifulsoup4
!pip install lxml
!pip install scrapy 
!pip install newspaper3k
!pip install googlesearch-python
!pip install yfinance
!pip install pytrends
!pip install transformers
!pip install datasets
!pip install sentence-transformers
!pip install bitsandbytes
!pip install peft
!pip install wandb
!pip install git+

#  importing all the necessary libraries to create the AI vision system
import requests
from PIL import Image
from io import BytesIO  
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np
import cv2
from deepface import DeepFace
from fer import FER
from mtcnn import MTCNN
import tensorflow as tf
from keras.models import load_model
import dlib
import face_recognition
import imutils
import mediapipe as mp
import imageio
import moviepy.editor as mp_editor
from gtts import gTTS

#The code that now generates the real a random face with a name and bio using free API's
import random
import string
import json
from IPython.display import display, HTML
import gradio as gr
import openai
import os
from dotenv import load_dotenv
from langchain import OpenAI, LLMChain  
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
import pdfplumber
from llama_index import (
    GPTVectorStoreIndex, SimpleDirectoryReader,
    LLMPredictor, PromptHelper, ServiceContext
)
from bs4 import BeautifulSoup
import lxml 
import scrapy
from newspaper import Article
from googlesearch import search
import yfinance as yf
from pytrends.request import TrendReq
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline
)
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import bitsandbytes as bnb
from peft import PeftModel  
import wandb
import git
import time
import re   
import sys
import math
import shutil
import threading
import asyncio
import concurrent.futures
from datetime import datetime
import base64
import hashlib
import uuid
import zipfile
import tarfile
import subprocess
import platform
import inspect
import traceback
import logging
import warnings
import functools
import contextlib
import itertools    
import operator
import collections
import enum
import dataclasses
import typing
import pathlib
import tempfile
import shutil
import glob
import fnmatch
import sched
import signal
import psutil
import resource
import ctypes
import struct       
import socket
import ssl
import http.client
import urllib.request
import urllib.parse
import urllib.error
import http.server
import ftplib
import smtplib
import email                                
import imaplib
import poplib


# Load environment variables from .env file
load_dotenv()
#using huggingface API to generate the faces
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')
# Function to generate a random face using the Hugging Face API
def generate_random_face():
    url = "https://api-inference.huggingface.co/models/fffiloni/Realistic_Vision_V2.0"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": "a high quality photo of a person", "options": {"wait_for_model": True}}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        image_data = response.content
        image = Image.open(BytesIO(image_data))
        return image
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")
# Function to generate a random name using the Random User API
def generate_random_name():
    url = "https://randomuser.me/api/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        name = data['results'][0]['name']
        full_name = f"{name['first']} {name['last']}"
        return full_name
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")
# Function to generate a random bio using the Random User API
def generate_random_bio():
    url = "https://randomuser.me/api/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        bio = data['results'][0]['location']['street']
        full_bio = f"{bio['number']} {bio['name']}, {data['results'][0]['location']['city']}, {data['results'][0]['location']['country']}"
        return full_bio
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")
# Function to generate a new face with name and bio
def generate_new_face():
    face = generate_random_face()
    name = generate_random_name()
    bio = generate_random_bio()
    return face, name, bio
# Gradio interface to display the face, name, and bio
def display_face():
    face, name, bio = generate_new_face()
    plt.imshow(face)
    plt.axis('off')
    plt.title(f"Name: {name}\nBio: {bio}")
    plt.show()
    return face, name, bio
# Gradio interface                          
iface = gr.Interface(
    fn=display_face,
    inputs=[],
    outputs=[
        gr.Image(type="pil", label="Generated Face"),
        gr.Textbox(label="Name"),
        gr.Textbox(label="Bio")
    ],
    title="Random Face Generator",
    description="Generate a random face with a name and bio using free API's. Click the button to get a new face."
)
iface.launch()
# To run the code, simply execute the cell in Google Colab. Click the "Run" button in the Gradio interface to generate a new face with a name and bio.

#