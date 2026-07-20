import cv2
import speech_recognition as sr
import threading

# # print(sr.Microphone.list_microphone_names())

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

camera = cv2.VideoCapture(0)

r = sr.Recognizer()
r.energy_threshold = 300
r.dynamic_energy_threshold = True

running = True

def faceRecognition():
    global running
    while True:

        ret, frame = camera.read()

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        if len(eyes) > 0:
            cv2.putText(frame,
                        "Eyes detected",
                        (20,30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.7,
                        (0,0,255),
                        2)

        cv2.imshow("Detected Face", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            running = False
            break

    camera.release()
    cv2.destroyAllWindows()

def speechRecognition():
    while running:
        with sr.Microphone(device_index=1) as source:
            print("Speak...")
            r.adjust_for_ambient_noise(source, duration=0.3)
            audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            print("You said:", text)

        except sr.UnknownValueError:
            print("Didn't understand")

        except sr.RequestError as e:
            print(e)
        
faceR = threading.Thread(target=faceRecognition)
speechR = threading.Thread(target=speechRecognition)

print("Function started")

faceR.start()
speechR.start()

faceR.join()
speechR.join()

print("Function executed")