import pyscreenshot as ImageGrab
import time
from PIL import Image

for i in range(0,1):
	# print(i)
	time.sleep(5)
	im = ImageGrab.grab(bbox=(700,200,880,410))
	print ("saved....",i)
	im.save("D:\\My Project\\test\\"+str(i)+".PNG")
	#print ("Clear screen now and redraw now.... ")
	