import os
import streamlit as st
import logging
import requests
from PIL import Image ,ImageOps
from io import BytesIO
import numpy as np
import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
# st.write("""
#          # Rock-Paper-Scissor Hand Sign Prediction
#          """
#          )
# st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")
st.title('Defect Classification')
st.subheader(""" Upload an image and run classifiaction on it.\n""")
    #GLCM Technique
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    #img_rgb = cv2.imread(image);
    st.image(image, use_column_width=True)
    size = (150,150);
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = np.asarray(image)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    glcmMatrix=(greycomatrix(img_gray, [1], [0], levels=256))
    proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy'];
    properties =np.zeros((1,5))
    
    for j in range(0, len(proList)):
        properties[0,j]=(greycoprops(glcmMatrix, prop=proList[j]))
    #features = np.array([properties[0],properties[1],properties[2],properties[3],properties[4]]);
    #df = pd.DataFrame(features,columns=proList)
    filename = 'gclm_model.sav'; 
    neigh1 = pickle.load(open(filename, 'rb'));
    #features = features.reshape(-1, 1)
    testt1=neigh1.predict(properties);
    if testt1==1:
       st.write("crease") ;
    elif testt1== 2:
        st.write("crescent_gap");
    elif testt1 == 3:
        st.write("inclusion");
    elif testt1 == 4 :
       st.write("oil_spot");
    elif testt1 == 5:
        st.write("punching_hole");
    elif testt1 == 6:
        st.write("rolled_pit");
    elif testt1 == 7:
        st.write("silk_spot");
    elif testt1 == 8:
        st.write("waist folding");
    elif testt1 == 9:
        st.write("water_spot");
    else:
        st.write("welding_line");
    st.text("\n Probability ")
    st.write(neigh1.predict_proba(properties))
#img_rgb  = cv2.imread(st.file_uploader(label='upload image here!'));
# st.image(img_rgb, use_column_width=True)

