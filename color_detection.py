import cv2
import numpy as np

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges and their names
    colors = {
        "Red": [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],  # red has two ranges in HSV
        "Green": [(36, 50, 70), (89, 255, 255)],
        "Blue": [(94, 80, 2), (126, 255, 255)],
        "Yellow": [(15, 150, 150), (35, 255, 255)],
    }

    for color_name, bounds in colors.items():
        if color_name == "Red":
            # Red needs two masks
            lower1, upper1, lower2, upper2 = bounds
            mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = mask1 + mask2
        else:
            lower, upper = bounds
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # only consider significant areas
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show the result
    cv2.imshow("Color Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy windows
cap.release()
cv2.destroyAllWindows()
