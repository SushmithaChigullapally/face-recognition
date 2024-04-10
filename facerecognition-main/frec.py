

# # from time import sleep
# # import time
# # import cv2
# # import numpy as np
# # import face_recognition
# # import os
# # # import json

# # scale = 0.25    #to resize the input video frames to improve processing speed. 
# # box_multiplier = 1/scale  


# # # Define a videocapture object
# # cap = cv2.VideoCapture(0)

# # # Images and names
# # classNames = []
# # path = 'faces'

# # # Function for Find the encoded data of the input image
# # # Reading the training images and classes and storing into the corresponding lists
# # for img in os.listdir(path):
# #     classNames.append(os.path.splitext(img)[0])

# # # Find encodings of training images

# # encodes = open('faces.dat', 'rb')
# # knownEncodes = np.load(encodes)
# # print('Encodings Loaded Successfully')


# # consecutive_count = 0
# # max_consecutive_count = 5 # Adjust this value as needed
# # previous_name = None

# # while True:
# #     success, img = cap.read()  # Reading Each frame

# #    # Resize the frame
# #     Current_image = cv2.resize(img,(0,0),None,scale,scale)
# #     Current_image = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)

# #     # Find the face location and encodings for the current frame
    
# #     face_locations = face_recognition.face_locations(Current_image,  model='cnn')
# #     face_encodes = face_recognition.face_encodings(Current_image,face_locations)
# #     for encodeFace,faceLocation in zip(face_encodes,face_locations):
# #         # matches = face_recognition.compare_faces(knownEncodes,encodeFace, tolerance=0.5)
# #         matches = face_recognition.compare_faces(knownEncodes,encodeFace)
# #         faceDis = face_recognition.face_distance(knownEncodes,encodeFace)
# #         matchIndex = np.argmin(faceDis) #The smaller the Euclidean distance, the more similar the faces are.
# #         #Therefore, np.argmin(faceDis) is used to find the index of the smallest distance in the array.
# #         # If match found then get the class name for the corresponding match

# #         if matches[matchIndex]:
# #             name = classNames[matchIndex].upper()

# #         else:
# #             name = 'Unknown'
# #         print(name)
# #         if name == previous_name:
# #             consecutive_count += 1
# #             if consecutive_count >= max_consecutive_count:
# #                 print(f"Detected {max_consecutive_count} consecutive frames with the same face. Stopping.")
# #                 break
# #         else:
# #             consecutive_count = 0 
        
# #         previous_name = name
# #         y1,x2,y2,x1=faceLocation
# #         y1,x2,y2,x1=int(y1*box_multiplier),int(x2*box_multiplier),int(y2*box_multiplier), int(x1*box_multiplier)

# #         # Draw rectangle around detected face

# #         cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
# #         cv2.rectangle(img,(x1,y2-20),(x2,y2),(0,255,0),cv2.FILLED)
# #         cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)
# #     cv2.imshow("window_name", img)
# #     if cv2.waitKey(1) & 0xFF == ord('q') or consecutive_count >= max_consecutive_count:
# #         break
# #     # cv2.imwrite("filename.jpg", img)


# #     # sleep(5000)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break


# # #closing all open windows 
# # cap.release()
# # cv2.destroyAllWindows()

# from time import sleep
# import cv2
# import numpy as np
# import face_recognition
# import os
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler

# # Function to retrain when a new image is added
# def retrain():
#     images = []
#     classNames = []
#     path = 'faces'

#     # Reading the training images and classes and storing into the corresponding lists
#     for img in os.listdir(path):
#         image = cv2.imread(f'{path}/{img}')
#         images.append(image)
#         classNames.append(os.path.splitext(img)[0])

#     print(classNames)

#     encodeList = []
#     for i in range(len(images)):
#         img = images[i]
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # Fix the index error by using modulo to wrap around
#         encode = face_recognition.face_encodings(img)[i % len(face_recognition.face_encodings(img))]
#         encodeList.append(encode)

#     faceData = open('faces.dat', 'wb')
#     np.save(faceData, encodeList)
#     print("Training Completed")

#     return classNames  # Return the classNames list

# # Event handler for file system changes
# class MyHandler(FileSystemEventHandler):
#     def on_modified(self, event):
#         if event.is_directory and os.path.basename(event.src_path) == 'faces':
#             print(f"Change detected in the 'faces' folder. Retraining...")
#             global knownEncodes, classNames
#             classNames = retrain()  # Update classNames when retraining

# # Define a videocapture object
# cap = cv2.VideoCapture(0)

# # Load the known face encodings
# encodes = open('faces.dat', 'rb')
# knownEncodes = np.load(encodes)
# print('Encodings Loaded Successfully')

# consecutive_count = 0
# max_consecutive_count = 5  # Adjust this value as needed
# previous_name = None

# # Initialize classNames outside the loop
# classNames = retrain()

# # Watch for changes in the 'faces' folder
# event_handler = MyHandler()
# observer = Observer()
# observer.schedule(event_handler, path='.', recursive=False)
# observer.start()

# # Main loop
# while True:
#     success, img = cap.read()  # Reading Each frame

#     # Resize the frame
#     Current_image = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#     Current_image = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)

#     # Find the face location and encodings for the current frame
#     face_locations = face_recognition.face_locations(Current_image, model='cnn')
#     face_encodes = face_recognition.face_encodings(Current_image, face_locations)

#     for encodeFace, faceLocation in zip(face_encodes, face_locations):
#         matches = face_recognition.compare_faces(knownEncodes, encodeFace)
#         faceDis = face_recognition.face_distance(knownEncodes, encodeFace)
#         matchIndex = np.argmin(faceDis)

#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#         else:
#             name = 'Unknown'
#         print(name)
#         if name == previous_name:
#             consecutive_count += 1
#             if consecutive_count >= max_consecutive_count:
#                 print(f"Detected {max_consecutive_count} consecutive frames with the same face. Stopping.")
#                 break
#         else:
#             consecutive_count = 0

#         previous_name = name
#         y1, x2, y2, x1 = faceLocation
#         y1, x2, y2, x1 = int(y1 * 4), int(x2 * 4), int(y2 * 4), int(x1 * 4)

#         # Draw rectangle around detected face
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.rectangle(img, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv2.FILLED)
#         cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

#     cv2.imshow("window_name", img)
#     if cv2.waitKey(1) & 0xFF == ord('q') or consecutive_count >= max_consecutive_count:
#         break

# # Closing all open windows and stopping the observer
# cap.release()
# cv2.destroyAllWindows()
# observer.stop()
# observer.join()


from time import sleep
import time
import cv2
import numpy as np
import face_recognition
import os
import json 
import face_recognition, os, cv2
from tqdm import tqdm



images = []
classNames = []
path = 'faces'

# Function for Find the encoded data of the input image
# Reading the training images and classes and storing into the corresponding lists
for img in os.listdir(path):
    image = cv2.imread(f'{path}/{img}')
    images.append(image)
    classNames.append(os.path.splitext(img)[0])

print(classNames)
def encodeImages(images):
    encodeList = []
    for i in tqdm(range(len(images)), desc="Encoding ", ascii=False, ncols=50, colour='green', unit=' Files'):
        img = images[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        # print(encode)
        encodeList.append(encode)
    faceData = open('faces.dat', 'wb')
    np.save(faceData, encodeList)
    faceData = np.load('faces.dat')
    # print(faceData)
    return encodeList

encodeImages(images)
print("Encoding Completed")





# Define the path for training images for OpenCV face recognition Project

scale = 0.25    
box_multiplier = 1/scale
 

# Define a videocapture object
cap = cv2.VideoCapture(0)
 
# Images and names 
classNames = []
path = 'faces'

# Function for Find the encoded data of the input image
# Reading the training images and classes and storing into the corresponding lists
for img in os.listdir(path):
    classNames.append(os.path.splitext(img)[0])

# Find encodings of training images

encodes = open('faces.dat', 'rb')
knownEncodes = np.load(encodes)
print('Encodings Loaded Successfully')

while True:
    success, img = cap.read()  # Reading Each frame

   # Resize the frame
    Current_image = cv2.resize(img,(0,0),None,scale,scale)
    Current_image = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)

    # Find the face location and encodings for the current frame
    
    face_locations = face_recognition.face_locations(Current_image,  model='cnn')
    face_encodes = face_recognition.face_encodings(Current_image,face_locations)
    for encodeFace,faceLocation in zip(face_encodes,face_locations):
        matches = face_recognition.compare_faces(knownEncodes,encodeFace, tolerance=0.4)
        # matches = face_recognition.compare_faces(knownEncodes,encodeFace)
        faceDis = face_recognition.face_distance(knownEncodes,encodeFace)
        matchIndex = np.argmin(faceDis)


        # If match found then get the class name for the corresponding match

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

        else:
            name = 'Unknown'
        print(name)
        y1,x2,y2,x1=faceLocation
        y1,x2,y2,x1=int(y1*box_multiplier),int(x2*box_multiplier),int(y2*box_multiplier), int(x1*box_multiplier)

        # Draw rectangle around detected face

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-20),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)
    cv2.imshow("window_name", img)
    # sleep(5000)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#closing all open windows 

cap.release()
cv2.destroyAllWindows()