import cv2
import speech_recognition as sr
import threading
import mediapipe as mp

# # print(sr.Microphone.list_microphone_names())

print(mp.__version__)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

r = sr.Recognizer()
r.energy_threshold = 300
r.dynamic_energy_threshold = True

running = True
sentences=[]

def faceRecognition():
    global running
    while True:

        _, frame = camera.read()

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = mp_face_mesh.process(rgb)
        
        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]

        LEFT_EYE = [33, 133]
        RIGHT_EYE = [362, 263]

        if result.multi_face_landmarks:
            
            for faceLandmarks in result.multi_face_landmarks:
                landmarks = faceLandmarks.landmark
                h, w, _ = frame.shape

                for idx in LEFT_IRIS:
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                for idx in RIGHT_EYE:
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)

                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                mp_draw.draw_landmarks(frame,faceLandmarks , mp.solutions.face_mesh.FACEMESH_TESSELATION)



        cv2.imshow("Detected Face", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            running = False
            break

    camera.release()
    cv2.destroyAllWindows()

def speechRecognition():
    with sr.Microphone(device_index=1) as source:
        while running:
            print("Speak...")
            r.adjust_for_ambient_noise(source, duration=0.3)
            audio = r.listen(source)

            try:
                text = r.recognize_google(audio)
                sentences.append(text)
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

print(sentences)
print("Function executed")