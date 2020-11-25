import numpy as np
import cv2
import os
import ssl
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nClasses = len(classes)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=2500, train_size=7500, random_state=9)
X_train_scalled = X_train/255.0
X_test_scalled = X_test/255.0


classifier = LogisticRegression(solver='saga', multi_class='multinomial')
classifier.fit(X_train_scalled, y_train)

y_pred = classifier.predict(X_test_scalled)

print("Accuracy : ", accuracy_score(y_test, y_pred))


def take_snapshot():
    videoCaptureObject = cv2.VideoCapture(1)
    result = True
    while result:
        try:
            ret, frame = videoCaptureObject.read()
            greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = greyFrame.shape
            upperLeft = (int(width/2-56), int(height/2-56))
            bottomRight = (int(width/2+56), int(height/2+56))
            cv2.rectangle(greyFrame, upperLeft, bottomRight, (0, 0, 255), 2)
            # cv2.imwrite("webCamImage.jpg", frame)
            # roi=region of interest
            roi = greyFrame[upperLeft[1]:bottomRight[1],
                            upperLeft[0]:bottomRight[0]]
            impill = Image.fromArray(roi)
            imBW = impill.convert('L')
            imBWResized = imBW.resize((28, 28), Image.ANTIALIAS)
            imFlipped = Pill.Image.invert(imBWResized)
            pixelFilter = 20
            minPixel = np.percentile(imFlipped, pixelFilter)
            imScaled = np.clip(imFlipped-minPixel, 0, 255)
            maxPixel = np.max(imFlipped)
            imFlippedScaled = np.asarray(imScaled/maxPixel)
            testSample = np.array(imFlippedScaled).reshape(1, 784)
            testPred = classifier.predict(testSample)
            print("predicted class is: ", testPred)

            cv2.imshow("frame", gray)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            result = False

        except Exception as e:
            pass

    videoCaptureObject.release()
    cv2.destroyAllWindows()


take_snapshot()
