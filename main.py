
from tkinter import filedialog
import face_recognition
import cv2
import os


face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
directory_path = 'database/'
faces_dict = {}
count = 0

for filename in os.listdir(directory_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        person_name = os.path.splitext(filename)[0]         # Remove the file extension to get the person's name
        file_path = os.path.join(directory_path, filename)
        faces_dict[person_name] = file_path                 # Add to the dictionary

file_path = filedialog.askopenfilename()
while not file_path:
    file_path = filedialog.askopenfilename()
live = cv2.imread(file_path)
gray = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  # white color
thickness = 2

for (k, v) in faces_dict.items():
    first = face_recognition.load_image_file(v)
    try:
        first = face_recognition.face_encodings(first)[0]
    except IndexError as e:
        print('Face not detected in the first image')

    if faces is ():
        print("No faces found")
    else:
        for (x, y, w, h) in faces:
            face = live[y:y+h, x:x+w]
            cv2.imwrite('face.jpg', face)
            face = face_recognition.load_image_file("face.jpg")
            try:
                second = face_recognition.face_encodings(face)[0]
                if os.path.exists('face.jpg'):
                    os.remove('face.jpg')
            except IndexError as e:
                print('Face not detected in the second image')
            try:
                result = face_recognition.compare_faces([first], second)
                if result[0]:
                    count += 1
                    cv2.rectangle(live, (x, y), (x + w, y + h), (127, 0, 255), 2)
                    text_position = (x, y - 10)

                    text_size = cv2.getTextSize(k, font, font_scale, thickness)[0]
                    text_x = x
                    text_y = y - 10
                    bg_x = x
                    bg_y = y - text_size[1] - 10

                    # Draw a filled rectangle (background for the text)
                    cv2.rectangle(live, (bg_x, bg_y), (bg_x + text_size[0], bg_y + text_size[1] + 5), (0, 0, 0), -1)

                    cv2.putText(live, k, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            except NameError:
                print('Please try to load better image with better face view')
if count == 0:
    print("No face was recognized.")
elif count == 1:
    print("1 face was recognized.")
else:
    print(f'{count} faces were recognized.')
cv2.imwrite("result.jpg", live)