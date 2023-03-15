import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((4 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:4, 0:6].T.reshape(-1, 2)

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(0)
for i in range(50):
    ret, img = cap.read()

for i in range(10):
    while True:
        ret, img = cap.read()
        cv2.imshow('img', img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("Capture")
            break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (4, 6), None)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)

cap.release()
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx)
