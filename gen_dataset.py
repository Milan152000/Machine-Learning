import csv
import cv2,os
import numpy as np

label = "9"

dirList = os.listdir("D:\\My Project\\orig_images\\9\\901\\")
print(dirList)
for img in dirList:
	data=[]
	im = cv2.imread("D:\\My Project\\orig_images\\9\\901\\"+str(img))
	im_gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY)
	im_gray = cv2.GaussianBlur(im_gray, (15,15), 0)
	roi = cv2.resize(im_gray, (28,28), interpolation=cv2.INTER_AREA)

	data.append(label)
	rows,cols = roi.shape

	for i in range(rows):
		for j in range(cols):
			k = roi[i,j]
			if k>175:
				k=0
			else:
				k=1

			data.append(k)

# data = ["label"]
# for i in range(0,784):
# 	data.append("pixel"+str(i))
			
	with open("D:\\My Project\\csv\\dataset.csv",'a',newline='') as f:
		writer = csv.writer(f)
		writer.writerow(data)
	


