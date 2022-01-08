import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os
import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Model, model_from_json
import time
from bokeh.models.widgets import Div
import streamlit as st
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image,ImageEnhance

# class
class FaEmoModel(object):
    
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        #self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        global session
        #set_session(session)
        self.preds = self.loaded_model.predict(img)
        return FaEmoModel.EMOTIONS_LIST[np.argmax(self.preds)]
    def predict_emotion1(self, img):
        global session
        #set_session(session)
        self.preds = self.loaded_model.predict(img)
        return FaEmoModel.EMOTIONS_LIST[int(np.argmax(self.preds))]    

#importing the cnn model+using the CascadeClassifier to use features at once to check if a window is not a face region
st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FaEmoModel("emotion_model1.json", "emotion_model1.h5")
#model = model_from_json(open("model.json", "r").read())
#model.load_weights('best_accuracy_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX 




#facial expressions detecting function
def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:

			fc = gray[y:y+h, x:x+w]
			roi = cv2.resize(fc, (48, 48))
			pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
			cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
			return img,faces,pred 



class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = model.predict_emotion1(roi)
                output = str(prediction)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img




def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    activities = ["Home", "Upload Picture","Webcam Face Detection"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown(
        """ Developed by Vinay    
            Email : vkvinay5988@gmail.com  
            """)
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.
                 1. Face detection.
                 2. Face emotion recognition.



                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)



    elif choice == 'Upload Picture':
        #html_choice = """ 
        #<marquee behavior="scroll" direction="left" width="100%;">
        #<h2 style= "color: #000000; font-family: 'Raleway',sans-serif; font-size: 44px; font-weight: 700; line-height: 102px; margin: 0 0 24px; text-align: center; text-transform: uppercase;">Facial Expression Recognition Web Application </h2>
        #</marquee><br>
        #"""
        #st.markdown(html_choice, unsafe_allow_html=True)

        st.subheader(":smile: :hushed: :worried: :rage: :fearful:")
        image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

        #if image if uploaded, display the progress bar and the image
        if image_file is not None:
            our_image = Image.open(image_file)
            st.markdown("**Original Image**")
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)
            st.image(our_image)
        if image_file is None:
            st.error("No image uploaded yet")
            return

        # Face Detection
        task = ["Faces"]
        feature_choice = st.sidebar.selectbox("Find Features", task)
        if st.button("Process"):
            if feature_choice == 'Faces':
                st.markdown("**Processing...\n**")

                #Progress bar
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)
                    progress.progress(i+1)
                #End of Progress bar

                result_img,result_faces,prediction = detect_faces(our_image)
                if st.image(result_img):
                    st.success("Found {} faces".format(len(result_faces)))

                    if prediction == 'Happy':
                        st.subheader("YeeY! You look **_Happy_** :smile: today, always be! ")
                    elif prediction == 'Angry':
                        st.subheader("You seem to be **_Angry_** :rage: today, just take it easy! ")
                    elif prediction == 'Disgust':
                        st.subheader("You seem to be **_Disgusted_** :rage: today! ")
                    elif prediction == 'Fear':
                        st.subheader("You seem to be **_Fearful_** :fearful: today, be couragous! ")
                    elif prediction == 'Neutral':
                        st.subheader("You seem to be **_Neutral_** today, wish you a happy day! ")
                    elif prediction == 'Sad':
                        st.subheader("You seem to be **_Sad_** :worried: today, smile and be happy! ")
                    elif prediction == 'Surprise':
                        st.subheader("You seem to be **_Surprised_** today! ")
                    else :
                        st.error("Your image does not seem to match the training dataset's images! Please try another image!")
                    


main()
