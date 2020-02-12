
   # Voice Biometrics Authentication and Face Recognition<br>

Voice Biometrics Authentication using GMM and Face Recognition Using Facenet and Tensorflow
___

   ## How to Run :
   
 **Install dependencies by running**  For Linux Terminal : ```pip3 install -r requirement.txt``` 
 ``` pip install -r requirements.txt```
 
 ### 1.Run in Jupyter notebook - main.ipynb
 ___
 
 ### 2.Run in terminal in following way :
 
   **To add new user :**
   ```
     python3 add_user.py
   ```
   **To Recognize user :**
   ```
     python3 recognize.py
   ```
   **To Recognize until KeyboardInterrupt (ctrl + c) by the user:**
   ```
     python3 recognize_until_keyboard_Interrupt.py
   ```
   **To delete an existing user :**
   ```
     python3 delete_user.py
   ```
___
## Voice Authentication

   *For Voice recognition, **GMM (Gaussian Mixture Model)** is used to train on extracted MFCC features from audio wav           file.*<br><br>
     
## Face Recognition

  *Face Recognition system using **Siamese Neural network**. The model is based on the **FaceNet model** implemented using Tensorflow and OpenCV implementaion has been done for realtime face detection and recognition.<br>
The model uses face encodings for identifying users.<br><br>*
The program uses a python dictionary for mapping for users to their corresponding face encodings. <br>
___
<br>


**Controlling the face recognition accuracy:**
 The threshold value controls the confidence with which the face is recognized, you can control it by changing the value which is here 0.5. <br><br>
 
___ 
**Another version of recognizing user will keep runnning until KeyboardInterrupt by the user. It is a modified version of recognize() function for real time situations.**
 ___
**References :**
*Code for Facenet model is based on the assignment from Convolutional Neural Networks Specialization by Deeplearning.ai on Coursera*.<br>
https://www.coursera.org/learn/convolutional-neural-networks/home/welcome 
*Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)*
*The pretrained model used is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.*
*Inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet*
