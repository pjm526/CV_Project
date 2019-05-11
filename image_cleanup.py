import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def gaussian_method(img_p):

	img = img_p.copy()
	result_img_g_temp = img.copy()

	l = len(img)
	b = len(img[0])

	print("l " + str(l))
	print("b " + str(b))


	for i in range(l):
		for j in range(17,b-17):
			result_img_g_temp[i][j] = ((0.0)*img[i][j-17]) + ((0.0)*img[i][j-16]) + ((0.0)*img[i][j-15]) + ((0.0)*img[i][j-14]) + \
			((0.0)*img[i][j-13]) + ((0.0)*img[i][j-12]) + ((0.0)*img[i][j-11]) + ((0.0)*img[i][j-10]) + \
			((0.0)*img[i][j-9]) + ((0.0)*img[i][j-8]) + ((0.0)*img[i][j-7]) + ((0.0)*img[i][j-6]) + \
			((0.000003)*img[i][j-5]) + ((0.000229)*img[i][j-4]) + ((0.005977)*img[i][j-3]) + \
			((0.060598)*img[i][j-2]) + ((0.24173)*img[i][j-1]) + ((0.382925)*img[i][j]) + \
			((0.24173)*img[i][j+1]) + ((0.060598)*img[i][j+2]) + ((0.005977)*img[i][j+3]) + \
			((0.000229)*img[i][j+4]) + ((0.000003)*img[i][j+5]) + \
			((0.0)*img[i][j+6]) + ((0.0)*img[i][j+7]) + ((0.0)*img[i][j+8]) + \
			((0.0)*img[i][j+9]) + ((0.0)*img[i][j+10]) + ((0.0)*img[i][j+11]) + \
			((0.0)*img[i][j+12]) + ((0.0)*img[i][j+13]) + ((0.0)*img[i][j+14]) + \
			((0.0)*img[i][j+15]) + ((0.0)*img[i][j+16]) + ((0.0)*img[i][j+17])

	result_img_g = result_img_g_temp.copy()
	for j in range(b):
		for i in range(17,l-17):
			result_img_g[i][j] = ((0.0)*img[i-17][j]) + ((0.0)*img[i-16][j]) + ((0.0)*img[i-15][j]) + ((0.0)*img[i-14][j]) + \
			((0.0)*img[i-13][j]) + ((0.0)*img[i-12][j]) + ((0.0)*img[i-11][j]) + ((0.0)*img[i-10][j]) + \
			((0.0)*img[i-9][j]) + ((0.0)*img[i-8][j]) + ((0.0)*img[i-7][j]) + ((0.0)*img[i-6][j]) + \
			((0.000003)*img[i-5][j]) + ((0.000229)*img[i-4][j]) + ((0.005977)*img[i-3][j]) + \
			((0.060598)*img[i-2][j]) + ((0.24173)*img[i-1][j]) + ((0.382925)*img[i][j]) + \
			((0.24173)*img[i+1][j]) + ((0.060598)*img[i+2][j]) + ((0.005977)*img[i+3][j]) + \
			((0.000229)*img[i+4][j]) + ((0.000003)*img[i+5][j]) + \
			((0.0)*img[i+6][j]) + ((0.0)*img[i+7][j]) + ((0.0)*img[i+8][j]) + \
			((0.0)*img[i+9][j]) + ((0.0)*img[i+10][j]) + ((0.0)*img[i+11][j]) + \
			((0.0)*img[i+12][j]) + ((0.0)*img[i+13][j]) + ((0.0)*img[i+14][j]) + \
			((0.0)*img[i+15][j]) + ((0.0)*img[i+16][j]) + ((0.0)*img[i+17][j])

	return result_img_g






def otsu_method(img_p):

	img = img_p.copy()

	l = len(img)
	b = len(img[0])
	H = {}
	for i in range(256):
		H[float(i)] = 0

	for i in range(l):
		for j in range(b):
			if img[i][j] in H:
				H[img[i][j]] = H[img[i][j]] + 1
			else:
				H[img[i][j]] = 1



	w0 = {}
	w1 = {}
	u0 = {}
	u1 = {}

	for h in sorted(H.keys()):
		p0 = 0
		p1 = 0
		up0 = 0
		up1 = 0
		for g in sorted(H.keys()):
			if g < h :
				p0 = p0 + (H[g]/(l*b))
				up0 = up0 + (g * (H[g]/(l*b)))
			else :
				p1 = p1 + (H[g]/(l*b))
				up1 = up1 + (g * (H[g]/(l*b)))
		w0[h] = p0
		w1[h] = p1
		if w0[h] != 0:
			u0[h] = up0/w0[h]
		if w1[h] != 0:
			u1[h] = up1/w1[h]





	thresold = 0
	sigmab2 = 0

	for i in range(256):
		if (w0[float(i)] != 0 and w1[float(i)] != 0):
			new_sigmab2 = w0[float(i)] * w1[float(i)] * pow( (u0[float(i)] - u1[float(i)]),2) 
			if new_sigmab2 > sigmab2 :
				sigmab2 = new_sigmab2
				thresold = i

	result_img = img.copy()

	for i in range(l):
		for j in range(b):
			total_value = img[i][j] 
			if total_value > thresold:
				result_img[i][j] = 255
			else:
				result_img[i][j] = 0

	return result_img

