import os
import streamlit as st
import logging
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops
import pickle
import pandas as pd
# st.write("""
#          # Rock-Paper-Scissor Hand Sign Prediction
#          """
#          )
# st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")
st.title('Defect Classification')
st.subheader(""" Upload an image and run classifiaction on it.\n""")
    #GLCM Technique
img_rgb  = cv2.imread(st.file_uploader(label='upload image here!'));
st.image(img_rgb, use_column_width=True)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY);
glcmMatrix=(greycomatrix(img_gray, [1], [0], levels=256))
proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy'];
for j in range(0, len(proList)):
    properties[j]=(greycoprops(glcmMatrix, prop=proList[j]))
features = np.array([properties[0],properties[1],properties[2],properties[3],properties[4]]);
filename = 'gclm_model.sav'; 
neigh1 = pickle.load(open(filename, 'rb'));
testt1=neigh1.predict(features);
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