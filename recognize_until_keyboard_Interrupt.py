import pyaudio
import wave
import cv2
import os
import pickle
import time
from scipy.io.wavfile import read
from IPython.display import Audio, display, clear_output

from main_functions import *

def recognize():
    # Voice Authentication
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    FILENAME = "./test.wav"
    try:
        while True:
            audio = pyaudio.PyAudio()

            # start Recording
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

            print("recording...")
            frames = []

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            print("finished recording")


            # stop Recording
            stream.stop_stream()
            stream.close()
            audio.terminate()

            # saving wav file 
            waveFile = wave.open(FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()

            modelpath = "./gmm_models/"

            gmm_files = [os.path.join(modelpath,fname) for fname in 
                        os.listdir(modelpath) if fname.endswith('.gmm')]

            models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]

            speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
                        in gmm_files]

            if len(models) == 0:
                print("No Users in the database!!")
                break

            #read test file
            sr,audio = read(FILENAME)

            # extract mfcc features
            vector = extract_features(audio,sr)
            log_likelihood = np.zeros(len(models)) 

            #checking with each model one by one
            for i in range(len(models)):
                gmm = models[i]         
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()

            pred = np.argmax(log_likelihood)
            identity = speakers[pred]

            # if voice not recognized than terminate the process
            if identity == 'unknown':
                    print("Not Recognized! Try again...")
                    time.sleep(1.5)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
            
            print( "Recognized as - ", identity)

             # face recognition
            print("Keep Your face infront of the camera")
            cap = cv2.VideoCapture(0)
            cap.set(3, 640)
            cap.set(4, 480)
            img_path = './saved_image/2.jpg'

            cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

            #loading the database 
            database = pickle.load(open('face_database/embeddings.pickle', "rb"))

            time.sleep(1.0)

            start_time = time.time()

            while True:
                curr_time = time.time()

                _, frame = cap.read()
                frame = cv2.flip(frame, 1, 0)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                face = cascade.detectMultiScale(gray, 1.3, 5)

                name = 'unknown'
                
                if len(face) == 1:

                    for (x, y, w, h) in face:
                        roi = frame[y-10:y+h+10, x-10:x+w+10]

                        fh, fw = roi.shape[:2]
                        min_dist = 100 

                        #make sure the face is of required height and width
                        if fh < 20 and fh < 20:
                            continue

                        #resizing image as required by the model
                        img = cv2.resize(roi, (96, 96))
        
                        #128 d encodings from pre-trained model
                        encoding = img_to_encoding(img)

                        # loop over all the recorded encodings in database 
                        for knownName in database:
                            # find the similarity between the input encodings and recorded encodings in database using L2 norm
                            dist = np.linalg.norm(np.subtract(database[knownName], encoding) )
                            # check if minimum distance or not
                            if dist < min_dist:
                                min_dist = dist
                                name = knownName

                    # if min dist is less then threshold value and face and voice matched than unlock the door
                    if min_dist <= 0.4 and name == identity:
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print ("Door Unlocked! Welcome " + str(name))
                        time.sleep(1.5)
                        break

                #open the cam for 3 seconds
                if curr_time - start_time >= 3:
                    break

                cv2.waitKey(1)
                cv2.imshow('frame', frame)

            cap.release()
            cv2.destroyAllWindows()

            if len(face) == 0:
                print('There was no face found in the frame. Try again...')
                continue

            elif len(face) > 1:
                print("More than one faces found. Try again...")
                continue

            elif min_dist > 0.4 or name != identity:
                print("Not Recognized! Try again...")
                continue

    except KeyboardInterrupt:
        print("Stopped")
        pass
    
if __name__ == '__main__':
    recognize()
