import cv2
import matplotlib 
import numpy as np

def readImagesAndTimes():
  times = np.array([ 1/1000.0, 1/400.0, 1/320.0, 1/80.0, 1/60.0, 1/4.0, 1/3.0, 0.8, 1, 3.2, 4, 10, 13], dtype=np.float32)

  filenames = ["img13.jpg", "img12.jpg", "img11.jpg", "img10.jpg", "img09.jpg", "img08.jpg", "img07.jpg", "img06.jpg", "img05.jpg", "img04.jpg", "img03.jpg", "img02.jpg", "img01.jpg"]
  images = []
  for filename in filenames:
    im = cv2.imread(filename)
    images.append(im)
    print(im.shape)
  return images, times


images, times = readImagesAndTimes()
#print(len(images))
#for i in range(13):
#	print(images[i].shape)

#Align by mtb
alignMTB = cv2.createAlignMTB()
alignMTB.process(images, images)

#print(alignMTB)

calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)

# 将图像合并为HDR线性图像
mergeDebevec = cv2.createMergeDebevec()
hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
# 保存图像
cv2.imwrite("hdrDebevec.hdr", hdrDebevec)