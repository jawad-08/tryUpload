import cv2 as cv
import time

# Load face detection model
face_net = cv.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
face_conf_threshold = 0.8

# Load age and gender prediction models
age_net = cv.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
gender_net = cv.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

# List of age and gender categories
age_categories = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-32)', '(33-45)', '(46-59)', '(60-100)']
gender_categories = ['Male', 'Female']

# Open a video capture
cap = cv.VideoCapture(0)
padding = 20

# Initialize variables for smoothing predictions
prev_age = None
prev_gender = None
consecutive_count = 0

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detctions.shape(2))
        confidence = detections[0, 0, i, 2]
        if confidence > face_conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame.shape[1])
            y1 = int(detections[0, 0, i, 4] * frame.shape[0])
            x2 = int(detections[0, 0, i, 5] * frame.shape[1])
            y2 = int(detections[0, 0, i, 6] * frame.shape[0])

            # Extract face ROI
            face_roi = frame[max(0, y1 - padding):min(y2 + padding, frame.shape[0] - 1),
                             max(0, x1 - padding):min(x2 + padding, frame.shape[1] - 1)]
            
            # Check if face ROI is empty or too small
            if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                continue

            # Perform age estimation
            age_blob = cv.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                             swapRB=False)
            age_net.setInput(age_blob)
            age_preds = age_net.forward()
            age_idx = age_preds[0].argmax()
            age = age_categories[age_idx]

            # Perform gender estimation
            gender_blob = cv.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                                swapRB=False)
            gender_net.setInput(gender_blob)
            gender_preds = gender_net.forward()
            gender_idx = gender_preds[0].argmax()
            gender = gender_categories[gender_idx]

            # Smooth predictions
            if age == prev_age and gender == prev_gender:
                consecutive_count += 1
            else:
                consecutive_count = 0

            if consecutive_count >= 10:
                # Display age and gender information
                label = f"Gender: {gender}, Age: {age}"
                cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
                prev_age = age
                prev_gender = gender

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv.imshow("Age Gender Demo", frame)

    # Check for key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv.destroyAllWindows()