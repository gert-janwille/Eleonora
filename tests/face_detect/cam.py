import cv2

number_of_detection = 5
process_this_frame = True


faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)

counter = 0;

while process_this_frame:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # flip the frame
    frame = cv2.flip(frame,1)
    
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert to greyscale
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Faces
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        if counter >= 20:
            counter = 20
        else:
            counter = counter + 1
    elif len(faces) == 0:
        counter = 0
    elif counter != 0:
        counter = counter - 1
    else:
        counter = 0
    
    print("Faces Detect:", len(faces))
    
    for idx, f in enumerate(faces):
        (x, y, w, h) = f

        # print(idx)
        
        if counter >= number_of_detection:
            cv2.rectangle(gray, (x, y), (x + w, y + h),(255,255,0), thickness=2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(gray, "Face Detected", (x + 6, y - 6), font, .5, (255, 255, 255), 1)

    
        if counter == number_of_detection:
            # Save just the rectangle faces in SubRecFaces
            sub_face = gray[y:y+h, x:x+w]
            FaceFileName = "./faces/face_" + str(y) + ".jpg"
            cv2.imwrite(FaceFileName, sub_face)
        
        
        # Eye Tracking
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = gray[y:y+h, x:x+w]
        # 
        # eyes = eyeCascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
    
    # Display the resulting frame
    cv2.imshow('Video', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
