import cv2
import time
import imutils

camera = cv2.VideoCapture(0)  # Use camera index 0 for default camera
time.sleep(1)

first_Frame = None
area = 500
threshold_value = 25  # Adjust this value as needed

while True:
    ret, img = camera.read()
    text = "normal"
    img = imutils.resize(img, width=500)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray_img, (21, 21), 0)

    if first_Frame is None:
        first_Frame = gaussian
        continue

    imgDiff = cv2.absdiff(first_Frame, gaussian)
    thresh = cv2.threshold(imgDiff, threshold_value, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)  # Grab contours from OpenCV 4+ return value

    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        text = "Moving object is detected"

        # Print contour area
        contour_area = cv2.contourArea(c)
        print(f"Contour area: {contour_area}")

    print(text)

    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Camera Feed", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()



