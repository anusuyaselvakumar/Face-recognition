import cv2
from sample_facerec import SimpleFacerec

sfr = SimpleFacerec()  # Encode from folder
sfr.load_encoding_images("images/")

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    
    #Detect faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations,face_names):
        top, left, bottom, right = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        
        cv2.putText(frame,name,(top,left-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,200),2)        
        cv2.rectangle(frame,(top,left),(bottom,right),(0,0,200),2)
    
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
