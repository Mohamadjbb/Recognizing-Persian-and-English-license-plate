import numpy as np
import cv2
from PIL import Image
import streamlit as st
from io import BytesIO
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import streamlit.components.v1 as components
def plate_detect(image):
    model2 = YOLO('plate.pt')
    try:
        img = np.array(image)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = model2.predict(source=img_bgr)
        img1 = results[0].plot()
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = Image.fromarray(img1)
        results = model2.predict(source=img_bgr)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                license_plate = img_bgr[y1:y2, x1:x2]
                cv2.imwrite('lp.jpg', license_plate)
                break
        cv2.imwrite('lp_en.jpg', license_plate)
        image = license_plate
        gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 50
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in filtered_contours]
        bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[0], box[1]))

        for i, (x, y, w, h) in enumerate(bounding_boxes):
            roi = image[y:y+h, x:x+w]
            cv2.imwrite(f'letter_{i}.jpg', roi)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imwrite('lp.jpg', image)
        return(image)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 50
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in filtered_contours]
        bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[0], box[1]))

        for i, (x, y, w, h) in enumerate(bounding_boxes):
            roi = image[y:y+h, x:x+w]
            cv2.imwrite(f'letter_{i}.jpg', roi)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imwrite('lp.jpg', image)
        return(image)

import os
def letter_detect(model_path='alefba_model2.h5'):
  model = load_model(model_path)
  plate=[]
  alpha= ['','1','2','3','4','5','6','7','8','9','Alef','B','SIN','D','E','GH','H','H jimi','L','M','N','SAD','TA','V','Y']
  try:
    for i  in range (1,20):
      img = cv2.imread(f'letter_{i}.jpg', cv2.IMREAD_GRAYSCALE)
      img = cv2.resize(img, (128, 128))
      img_reshaped = img.reshape((1, 128, 128, 1))
      answers= model.predict(img_reshaped)
      ans= np.argmax(answers)
      if(len(plate)<2 and ans<9):
        ans +=1
        plate.append(alpha[ans])
      elif(len(plate)==8):
        break
      elif(len(plate)==2 and ans>=9):
        ans +=1
        plate.append(alpha[ans])
      elif(len(plate)>2 and ans<9):
        ans +=1
        plate.append(alpha[ans])
      else:
        continue
    for i in range(0,500):
        try:
          os.remove(f'letter_{i}.jpg')
        except:
           continue
    return(plate)
  except:
    for i in range(0,500):
        try:
          os.remove(f'letter_{i}.jpg')
        except:
           continue
    return(plate)
  
def letter_detect_en(image):
    img = np.array(image)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)
    detected_text = ""
    for (bbox, text, prob) in result:
        detected_text += text + " "
    
    return detected_text

import os
def letter_detect(model_path='alefba_model2.h5'):
  model = load_model(model_path)
  plate=[]
  alpha= ['','1','2','3','4','5','6','7','8','9','Alef','B','SIN','D','E','GH','H','H jimi','L','M','N','SAD','TA','V','Y']
  try:
    for i  in range (1,20):
      img = cv2.imread(f'letter_{i}.jpg', cv2.IMREAD_GRAYSCALE)
      img = cv2.resize(img, (128, 128))
      img_reshaped = img.reshape((1, 128, 128, 1))
      answers= model.predict(img_reshaped)
      ans= np.argmax(answers)
      if(len(plate)<2 and ans<9):
        ans +=1
        plate.append(alpha[ans])
      elif(len(plate)==8):
        break
      elif(len(plate)==2 and ans>=9):
        ans +=1
        plate.append(alpha[ans])
      elif(len(plate)>2 and ans<9):
        ans +=1
        plate.append(alpha[ans])
      else:
        continue
    for i in range(0,10):
      os.remove(f'letter_{i}.jpg')
    return(plate)
  except:
      
      for i in range(0,500):
        try:
          os.remove(f'letter_{i}.jpg')
        except:
           continue

       
      return(plate)
  
def letter_detect_en(adrees='lp_en.jpg'):
  import cv2
  import easyocr
  image_path = 'lp_en.jpg'
  image = cv2.imread(image_path)
  reader = easyocr.Reader(['en'])
  result = reader.readtext(image)
  for (bbox, text, prob) in result:
      temp= text
      for i in range(0,500):
        try:
          os.remove(f'letter_{i}.jpg')
        except:
           continue
  return (temp)

def add_bg_from_local():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color:  #280050 ;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local()
st.title("PLATE DETECTION APPLICATION")
uploaded_file = st.file_uploader("Please Enter Youre Image Car")

if uploaded_file is not None:
    st.write("Successfully Uploaded")
    img_data = uploaded_file.read()
    image = Image.open(uploaded_file)
    detect = plate_detect(image)
    st.image(detect)
    if detect is not None:
        bool = True
    else:
        bool = False
    if bool:
        option = st.radio("Select the plate",('پلاک ایران','English plate'))
        if(st.button("Done")):
            if(option=='پلاک ایران'):
                st.write("در حال تشخیص حروف...")
                plate_letter = letter_detect()
                str_fa = ''
                for i in plate_letter:
                    str_fa += i   
                st.title(str_fa)
            elif(option=='English plate'):
                st.write("Processing..")
                plate_letter = letter_detect_en() 
                st.title(plate_letter)