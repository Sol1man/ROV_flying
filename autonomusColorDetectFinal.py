import cv2
import numpy as np

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_DUPLEX
while True:

    isAvilable,frame = cap.read()
    screenHeight, screenWidth = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    low_blue = np.array([94,80,2])
    high_blue =np.array([127,255,255])
    mask = cv2.inRange(hsv_frame,low_blue,high_blue)
    output = cv2.bitwise_and(frame, frame, mask=mask)

    ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for ctr in ctrs:

        area = cv2.contourArea(ctr)
        if area > 900:
            x,y,h,w = cv2.boundingRect(ctr)
            cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),2)
            (cX,cY)= (int(x+(h/2)),int(y+(w/2)))
            centerX = str(cX)
            centerY = str(cY)
            cv2.circle(frame,(cX,cY),3,(255,0,0),3)
            print('center x:',cX)
            print('center y:',cY)

    if ( cX > 0.95 * screenWidth) and (cX < 0.05 * screenWidth ):
        cv2.putText(frame,"getting away...",(30, 70), font,0.9, (250, 120, 0),2)
    elif ( cX > 0.95 * screenWidth):
        cv2.putText(frame, "moving right...", (30, 70), font, 0.9, (250, 120, 0),2)
    elif (cX < 0.05 * screenWidth ):
        cv2.putText(frame, "moving left....", (30, 70), font, 0.9, (250, 120, 0),2)
    elif(cX > 0.25*screenWidth and cX < 0.75*screenWidth):
        cv2.putText(frame, "getting closer...", (30, 70), font, 0.9, (250, 120, 0),2)

    resImage = [640, 480]
    resScreen = [1920, 1080]

    cv2.imshow('mask', mask)
    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == 83 or key == 115:  # unicode of the 'S' and 's' keys
        break

print('h',screenHeight)
print('w',screenWidth)
cap.release()
cv2.destroyAllWindows()