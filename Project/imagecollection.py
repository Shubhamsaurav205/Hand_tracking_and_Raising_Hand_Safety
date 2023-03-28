
import os
import time
import cv2

IMAGES_PATH = os.path.join('prac', 'imgs') #set path where the image will save
labels = ['Hand']
number_imgs = 5

cap = cv2.VideoCapture(0)
# Loop through labels
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    
    # Loop through image range
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        # Webcam feed
        ret, frame = cap.read()
        # Naming out image path
        imgname = os.path.join(IMAGES_PATH, label+'.jpg')
        # Writes out image to file 
        cv2.imwrite(imgname, frame)
        
        # Render to the screen
        cv2.imshow('Image Collection', frame)
        
        # 5 second delay between captures
        time.sleep(5)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()