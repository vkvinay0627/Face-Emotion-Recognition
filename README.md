# Face-Emotion-Recognition


## Problem Statement:


# Introduction : 

The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms. Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms. One of many challenges is how to ensure quality learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live class scenario is yet an open-end challenge.
## Problem Statement:

We will solve the above-mentioned challenge by applying deep learning algorithms to live video data. The solution to this problem is by recognizing facial emotions.

## Attribute Information:
Facial expression recognition system is a computer-based technology and therefore, it uses algorithms to instantaneously detect faces, code facial expressions, and recognize emotional states. It does this by analyzing faces in images or video through computer powered cameras embedded in laptops, mobile phones, and digital signage systems, or cameras that are mounted onto computer screens. Facial analysis through computer powered cameras generally follows three steps:

**A. Face detection**

Locating faces in the scene, in an image or video footage.

**B. Facial Feature Detection**

Extracting information about facial features from detected faces. For example, detecting the shape of facial components or describing the texture of the skin in a facial area.

**C. Facial expression and emotion Classification**

Analyzing the movement of facial features and/or changes in the appearance of facial features and classifying this information into expression-interpretative categories such as facial muscle activations like smile or frown; emotion categories happiness or anger; attitude categories like (dis)liking or ambivalence

## Facial Feature Detection and Emotion Classification

The Haar Casscade detects face and those faces are then cropped and convert to gray images. These gray images further get converted into iamge aaray for processing. Four CNN blocks and 5 Dense block with dropout probabilities of 0.25 are used. Each block has Batch Normalization layer, CNN layer (3x3) kernel, activation Function layer (ReLU) and max pooling (2x2). Dataset which is used to train this DNN is FER 2013 dataset. It has images of all 7 class of emotion.

Hyper-parameter that were used are epochs = 48,batch_size = 225 and learning_rate = 0.001

## Results
![Screenshot (532)](https://user-images.githubusercontent.com/48415899/154789569-be5a42f7-b9a7-4c71-9e4a-704a9c353a5e.png)

## Conclusion

Our model shows accuracy of 68% on validation set and 80% on train set.

To access weblink please click on this link(streamlit share): [Streamlit](https://share.streamlit.io/vkvinay0627/face-emotion-recognition/main/app.py)

To access weblink please click on this link(Heroku share): [herokuapp](https://face-emot.herokuapp.com/)

![WhatsApp Image 2022-02-19 at 12 45 14 PM](https://user-images.githubusercontent.com/48415899/154791149-f643d85f-10cc-4f9a-b256-e35368b872ab.jpeg)
