import cv2
import numpy as np
import playsound
import os, random
import imutils
from threading import Thread
from keras.models import load_model
import operator

faceCascade = cv2.CascadeClassifier('dataset/haarcascade_frontalface_alt2.xml')

def Play_Music(path):
    print(path )
    playsound.playsound(path)

def emoji(exp):

    #if Ex==exp:
    a='emoji/'+exp+'.png'
    '''    
    else:
        t=random.choice(os.listdir("music/"+exp+"//")) #play random song
        a='emoji/'+exp+'.png'
        path='music/'+exp+'/'+t

        if path !="":
            t=Thread(target=Play_Music, args = [path])
            t.deamon=True
            t.start()'''

    vc = cv2.imread(a)
    return(vc)

video_capture = cv2.VideoCapture(0)
model = load_model('dataset/experssion_detector.hdf5')

target = ['angry','disgust','fear','happy','sad','surprise','neutral']
font = cv2.FONT_HERSHEY_SIMPLEX

stats = {'angry':0, 'disgust':0, 'fear':0, 'happy':0, 'sad':0, 'surprise':0, 'neutral':0}

#while True:
for i in range(10):
    # Capture frame-by-frame 
    ret, frame = video_capture.read()

    # Resize the Frame to improve speed
    frame = imutils.resize(frame, width=450)

    # Convert to Gray-Scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,
        minSize=(25, 25))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2,5)
        face_crop = frame[y:y+h,x:x+w]
        face_crop = cv2.resize(face_crop,(48,48))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_crop = face_crop.astype('float32')/255
        face_crop = np.asarray(face_crop)
        face_crop = face_crop.reshape(1, 1,face_crop.shape[0],face_crop.shape[1])
        result = target[np.argmax(model.predict(face_crop))]
        cv2.putText(frame,result,(x,y), font, 1, (200,100,0), 3, cv2.LINE_AA)
        frame1 = emoji(result)
        frame1 = imutils.resize(frame1, width=150)
        cv2.imshow('frame',frame)
        cv2.imshow('emoji', frame1)  # Display the resulting frame
        stats[result] += 1
    k=cv2.waitKey(10)

    if k==27 & 0xFF == ord('q'):
        break
      


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

        #for k in d.keys():
#    print("{} => {}".format(k, d[k]))
mood = max(stats.items(), key=operator.itemgetter(1))[0]
songs = []
for file in os.listdir("music/{}".format(mood)):
    if file.endswith(".mp3"):
        songs.append("music/{}/{}".format(mood, file))
song = random.choice(songs)
os.system("mpg123 " + songs[0]) 




