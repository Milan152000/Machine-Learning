import joblib,cv2
import numpy as np
import sklearn
import time,os

model = joblib.load("D:\\My Project\\model\\svm_0to9_model_linear")
dir_list = os.listdir("D:\\My Project\\orig_images\\live\\")
print(dir_list)

img_folder = "D:\\My Project\\live_test\\"
fout = open("testing_x","w+")
pred_file = open("pred",'w+')
# pred = []

for img in dir_list:
    im = cv2.imread("D:\\My Project\\orig_images\\live\\"+str(img))

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

    # Thresholding image
    ret,im_th = cv2.threshold(im_gray, 175, 255, cv2.THRESH_BINARY)

    roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)

    cv2.imwrite(img_folder + "segmented.png", roi)

    rows,cols = roi.shape

    X = []

    # #Add pixel one-by-one into data Array.
    for i in range(rows):
        for j in range(cols):
            k = roi[i, j]
            if k > 175:
                k = 0
            else:
                k = 1
            X.append(k)

    fout.write(str(X))
    predictions = model.predict([X])
    print("Prediction: ", predictions[0])
    pred_file.write(str(predictions[0]))
    cv2.putText(im, "Prediction:" + str(predictions[0]), (20, 20), 0, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # cv2.startWindowThread()
    cv2.namedWindow("Result",cv2.WINDOW_NORMAL)
    cv2.imshow("Result", im)
    cv2.waitKey(1000)

