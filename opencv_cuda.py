import sys
import time
import cv2

### VALUES
NUM_REPEAT = 10000

### Read source image
img_src = cv2.imread("resource/lena.jpg")
cv2.imshow('img_src', img_src)


### Run with CPU
time_start = time.time()
for i in range (NUM_REPEAT):
	img_dst = cv2.resize(img_src, (300, 300))
time_end = time.time()
print ("CPU = {0}".format((time_end - time_start) * 1000 / NUM_REPEAT) + "[msec]")
cv2.imshow('CPU', img_dst)


### Run with GPU
img_gpu_src = cv2.cuda_GpuMat()	# Allocate device memory only once, as memory allocation seems to take time...
img_gpu_dst = cv2.cuda_GpuMat()
time_start = time.time()
for i in range (NUM_REPEAT):
	img_gpu_src.upload(img_src)
	img_gpu_dst = cv2.cuda.resize(img_gpu_src, (300, 300))
	img_dst = img_gpu_dst.download()
time_end = time.time()
print ("GPU = {0}".format((time_end - time_start) * 1000 / NUM_REPEAT) + "[msec]")
cv2.imshow('GPU', img_dst)


key = cv2.waitKey(0)
cv2.destroyAllWindows()

print(cv2.cuda.getCudaEnabledDeviceCount())
