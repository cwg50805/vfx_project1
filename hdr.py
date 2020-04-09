import cv2
import matplotlib 
import numpy as np
import random
import sys
import math
from math import log2

def readfile():
	#times = np.array([ 1/1000.0, 1/400.0, 1/320.0, 1/80.0, 1/60.0, 1/4.0, 1/3.0, 0.8, 1, 3.2, 4, 10, 13], dtype=np.float32)
	#filenames = ["img13.jpg", "img12.jpg", "img11.jpg", "img10.jpg", "img09.jpg", "img08.jpg", "img07.jpg", "img06.jpg", "img05.jpg", "img04.jpg", "img03.jpg", "img02.jpg", "img01.jpg"]
	
	#times = np.array([ 1/60.0, 1/30.0, 1/50.0, 1/2.0, 1/4.0, 1/3.0, 1/8.0, 1/15.0, 1/13.0], dtype=np.float32)
	#filenames = ["hdr_images/devil9.jpg", "hdr_images/devil8.jpg", "hdr_images/devil7.jpg", "hdr_images/devil6.jpg", "hdr_images/devil5.jpg", "hdr_images/devil4.jpg", "hdr_images/devil3.jpg", "hdr_images/devil2.jpg", "hdr_images/devil1.jpg"]

	times = np.array([ 1/332.0, 1/256.0, 1/512.0, 1/664.0, 1/790.0, 1/1025.0, 1/1328.0, 1/1580.0, 1/2049.0, 1/2660.0, 1/3125.0, 1/4098.0], dtype=np.float32)
	filenames = ["pic_jpg/IMG_5400.jpg", "pic_jpg/IMG_5401.jpg", "pic_jpg/IMG_5402.jpg", "pic_jpg/IMG_5403.jpg", "pic_jpg/IMG_5404.jpg", "pic_jpg/IMG_5405.jpg", "pic_jpg/IMG_5406.jpg", "pic_jpg/IMG_5407.jpg", "pic_jpg/IMG_5408.jpg", "pic_jpg/IMG_5409.jpg", "pic_jpg/IMG_5410.jpg", "pic_jpg/IMG_5411.jpg"]
	images = []
	for filename in filenames:
		im = cv2.imread(filename)
		images.append(im)
    #print(im.shape)
	return images, times

def weight(intensity):
	if intensity <= 127:
		return intensity + 1
	return (256-intensity)

np.set_printoptions(threshold=sys.maxsize)
images, times = readfile()
#print(images[0])
l_images = len(images)
#print(len(images))
#for i in range(13):
#	print(images[i].shape)

#Align by mtb
#alignMTB = cv2.createAlignMTB()
#alignMTB.process(images, images)

'''
#By library

#print(alignMTB)

calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)

# 将图像合并为HDR线性图像
mergeDebevec = cv2.createMergeDebevec()
hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
# 保存图像
cv2.imwrite("hdrDebevec.hdr", hdrDebevec)


'''
#z_min = 0
#z_max = 255
#print(images[0].shape[2])
log_t = [log2(t) for t in times]
hdr_image = np.zeros(images[0].shape, dtype=np.float32)
for i in range(images[0].shape[2]):
	
	#Put the image with same color layer together
	value_rgb = []
	for j in range(l_images):
		value_rgb.append(images[j][:,:,i])
	#print(value_rgb[0].shape)

	num_pick = 100
	#Pick point with different intensity to fully construct the response curve
	zij = np.zeros((l_images,num_pick), dtype=np.uint8)
	for j in range(num_pick):
		x = random.randint(0, value_rgb[0].shape[0]-1)
		y = random.randint(0, value_rgb[0].shape[1]-1)

		for k in range(l_images):
			zij[k,j] = value_rgb[k][x,y]
	#if i==0 :print(zij.shape)

	#Construct log delta t
	#print(log_t)

	#initialize A and B
	A = np.zeros((l_images * num_pick + 254 + 1, num_pick + 256 ), dtype=np.float32)
	B = np.zeros((A.shape[0], 1), dtype=np.float32)

	#Make g(127) = 0
	g_127_place = l_images * num_pick 
	A[g_127_place, 127] = 1

	#Fill in A's downward
	lamba = 150
	for j in range(254):
		w_j = weight(j+1)
		A[g_127_place + j + 1, j] = 1 * lamba * w_j
		A[g_127_place + j + 1, j + 1] = -2 * lamba * w_j
		A[g_127_place + j + 1, j + 2] = 1 * lamba * w_j
	#print(A)

	#Fill in A's upward and B
	for j in range(l_images):
		for k in range(num_pick):
			z = zij[j,k]
			w_z = weight(z)
			A[j * num_pick + k, z] = 1 * w_z
			A[j * num_pick + k, 256 + k] = -1 * w_z
			B[j * num_pick + k, 0] = log_t[j] * w_z

	#print(B)
	inv_A = np.linalg.pinv(A)
	X = np.dot(inv_A, B)
	
	#g(0) ~ g(255) = X[0~255]
	g = X[0:256]
	#print(g)
	# Save Numpy array to csv
	np.savetxt('g.csv', g, delimiter=',', fmt='%f')

	#Construct radiance map
	hdr_channel = np.zeros((images[0].shape[0], images[0].shape[1]), dtype=np.float32)

	for j in range(hdr_channel.shape[0]):
		for k in range(hdr_channel.shape[1]):
			w_sum = 0
			lnEi = 0
			for m in range(l_images):
				z = images[m][j,k,i]
				w_z = weight(z)
				lnEi += (g[z]-log_t[m]) * w_z
				#if j%1000 == 0 :print(w_z)
				w_sum += w_z
			lnEi /= w_sum
			hdr_channel[j,k] = 2**lnEi

	np.savetxt('hdr_channel.csv', hdr_channel, delimiter=',', fmt='%f')

	#hdr_image[..., i] = cv2.normalize(src = hdr_channel, dst = hdr_channel, alpha=0, beta=1024, norm_type=cv2.NORM_MINMAX)
	hdr_image[..., i] = hdr_channel

#print(log_t)
#print(hdr_image)
cv2.imwrite("pic_station.hdr", hdr_image)



