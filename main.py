import cv2
import speech_recognition as sr
import threading
import mediapipe as mp
import time

print(mp.__version__)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

r = sr.Recognizer()
r.energy_threshold = 300
r.dynamic_energy_threshold = True

running = True
sentences = []

def faceRecognition():
    global running
    eye_contact_time = 0.0

    # Initialize timing
    prev_frame_time = time.time()
    # Create a thinner drawing spec for the mesh
    custom_drawing_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(255, 255, 255))
    while running:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = mp_face_mesh.process(rgb)
        
        # Calculate time passed since the last frame
        current_time = time.time()
        delta_time = current_time - prev_frame_time
        prev_frame_time = current_time
        
        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]

        is_looking = False

        if result.multi_face_landmarks:
            for faceLandmarks in result.multi_face_landmarks:
                landmarks = faceLandmarks.landmark
                
                # --- LEFT EYE CALCULATIONS ---
                left_x = sum([landmarks[idx].x for idx in LEFT_IRIS]) / 4
                left_y = sum([landmarks[idx].y for idx in LEFT_IRIS]) / 4

                # Left eye corners (133 is inner, 33 is outer in mirrored view)
                left_eye_inner = landmarks[133].x
                left_eye_outer = landmarks[33].x
                left_width = abs(left_eye_outer - left_eye_inner)
                
                # Calculate where the iris is horizontally (ratio from 0.0 to 1.0)
                left_ratio = (left_x - min(left_eye_inner, left_eye_outer)) / (left_width + 1e-6)

                # --- RIGHT EYE CALCULATIONS ---
                right_x = sum([landmarks[idx].x for idx in RIGHT_IRIS]) / 4
                right_y = sum([landmarks[idx].y for idx in RIGHT_IRIS]) / 4

                # Right eye corners (362 is inner, 263 is outer)
                right_eye_inner = landmarks[362].x
                right_eye_outer = landmarks[263].x
                right_width = abs(right_eye_outer - right_eye_inner)
                
                # Calculate right iris ratio
                right_ratio = (right_x - min(right_eye_inner, right_eye_outer)) / (right_width + 1e-6)

                # --- DETERMINE EYE CONTACT ---
                # A ratio around 0.5 means the iris is perfectly centered.
                # We use a threshold (0.4 to 0.6) to allow for slight head movements.
                if 0.40 < left_ratio < 0.65 and 0.40 < right_ratio < 0.65:
                    is_looking = True

                # --- DRAWING VISUALS ---
                # Convert normalized coordinates (0-1) to pixel coordinates (0-width/height)
                cx_l, cy_l = int(left_x * w), int(left_y * h)
                cx_r, cy_r = int(right_x * w), int(right_y * h)
                
                # Draw green circles if making eye contact, red if looking away
                color = (0, 255, 0) if is_looking else (0, 0, 255)
                cv2.circle(frame, (cx_l, cy_l), 3, color, -1)
                cv2.circle(frame, (cx_r, cy_r), 3, color, -1)

                # Draw the face mesh
                mp_draw.draw_landmarks(
    image=frame, 
    landmark_list=faceLandmarks, 
    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
    landmark_drawing_spec=None, # Hides the dots completely
    connection_drawing_spec=custom_drawing_spec # Draws thin white lines
)

        # Update the timer if eye contact is maintained
        if is_looking:
            eye_contact_time += delta_time
            cv2.putText(frame, "Eye Contact: YES", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Eye Contact: NO", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display total accumulated time
        cv2.putText(frame, f"Total Time: {eye_contact_time:.1f}s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Detected Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running = False
            break

    camera.release()
    cv2.destroyAllWindows()
    print(f"\nFinal Eye Contact Time: {eye_contact_time:.2f} seconds")

def speechRecognition():
    # Make sure to handle the thread exit gracefully by setting a timeout
    with sr.Microphone(device_index=1) as source:
        while running:
            print("Speak...")
            r.adjust_for_ambient_noise(source, duration=0.3)
            try:
                # Adding a timeout prevents the thread from hanging indefinitely when you press 'q'
                audio = r.listen(source, timeout=2, phrase_time_limit=5)
                text = r.recognize_google(audio)
                sentences.append(text)
                print("You said:", text)
            except sr.WaitTimeoutError:
                pass # This just lets the loop check the 'running' flag again
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

print("Sentences:", sentences)
print("Function executed")