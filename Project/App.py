
##Importing Libraries 
import torch
import numpy as np
import cv2
from time import time
import pygame 


class HandDetection:
    

    def __init__(self, capture_index, model_name):
        """
        Constructor of the HandDetection class. It initializes the object
        with the capture index and the model name.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        


    def get_video_capture(self):
        """
        Function to return the video capture object
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        # Function to Loads the model for hand detection
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Function to run inference on the input frame and get the class labels 
        and coordinates of the objects detected
        """
        
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        Function to convert class index to class label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Function to draw bounding boxes and class labels around the detected objects in the frame
        """
        pygame.mixer.init()# Initializing Pygame audio module
        y_coord = 100 # Setting the y-coordinate to draw a horizontal line
        self.sound = pygame.mixer.Sound('siren.mp3') # Loading the siren sound
        
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
                # cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
            if int(row[1]*y_shape)<=y_coord:
                cv2.putText(frame, 'Your hand is in danger!', (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #self.sound.play()

        return frame

    def __call__(self):
        """
        #Main method to start the hand detection process
        """
        # y_coord1==50 # Setting the y-coordinate to draw a horizontal line1
        y_coord = 100# Setting the y-coordinate to draw a horizontal line2
        cap = self.get_video_capture()
        assert cap.isOpened()
      
        while True:
          
            ret, frame = cap.read()
            assert ret
            # cv2.line(frame, (0, y_coord1), (frame.shape[1], y_coord1), (0, 0, 255), thickness=3)#Code for boundryline1
            cv2.line(frame, (0, y_coord), (frame.shape[1], y_coord), (0,0,255), thickness=2)#Code for boundryline1
            
            
            frame = cv2.resize(frame, (416,416))#Resize
            
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 1)
            
            #cv2.putText(frame, f'FPS: {int(fps)}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            cv2.imshow('Hand Detection', frame)
            
            if cv2.waitKey(2) == ord('q'):# if press "q" the codeexecution will break
                break
      
        cap.release()
        
        
# Create a new object and execute.
detector = HandDetection(capture_index=0, model_name='best.pt')##in object we provide a 0(for ral time videocapture),and modelname
detector()