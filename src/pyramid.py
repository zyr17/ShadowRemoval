import cv2
import numpy as np
import copy
FourNear = [[1,0],[-1,0],[0,1],[0,-1]]
ShadowGroundTruth = 128


def RemoveOneShadow(originalpng, shadowpng): # get gray image
	shadownum = []
	illunum = []
	for (i, j), num in np.ndenumerate(originalpng):
		if shadowpng[i][j] == 0:
			#not shadow
			illunum.append(originalpng[i][j])
		else:
			#shadow
			shadownum.append(originalpng[i][j])
	shadowstd = np.std(shadownum)
	illustd = np.std(illunum)
	lamida = illustd / shadowstd
	alpha = np.mean(illunum) - lamida * np.mean(shadownum)
	res = copy.deepcopy(originalpng)
	for (i, j), num in np.ndenumerate(originalpng):
		if shadowpng[i][j] != 0:
			#shadow
			temp = np.int32(res[i][j])
			temp = alpha + lamida * temp
			if temp > 255:
				temp = 255
			if temp < 0:
				temp = 0
			res[i][j] = np.uint8(temp)
	return res

def GenerateOneShadow(stx, sty):
	l = [[stx, sty]]
	lpos = 0
	oneshadowimg = copy.deepcopy(allblackimg)
	while True:
		x = l[lpos][0]
		y = l[lpos][1]
		#if lpos % 100 == 0:
		#print lpos, x, y, shadowimg[x][y]
		#raw_input()
		if shadowimg[x][y] > ShadowGroundTruth:
			shadowimg[x][y] = 0
			oneshadowimg[x][y] = 255
			for i in xrange(4):
				X = x + FourNear[i][0]
				Y = y + FourNear[i][1]
				if X < 0 or Y < 0 or X >= shadowimg.shape[0] or Y >= shadowimg.shape[1]:
					continue
				if shadowimg[X][Y] > ShadowGroundTruth:
					l.append([X, Y])
		lpos += 1
		if lpos >= len(l):
			break
	return oneshadowimg

def ShadowRemoval(testdataset, testimgname, savepath, resultname):
	img = cv2.imread('../data/' + testdataset + '/original/' + testimgname + '.jpg',0)
	global shadowimg
	shadowimg = cv2.imread('../data/' + testdataset + '/groundtruth/' + testimgname + '.png', 0)
	cv2.imwrite(savepath + 'shadow.png', shadowimg)
	global allblackimg
	allblackimg	= copy.deepcopy(shadowimg)
	for (i, j), num in np.ndenumerate(allblackimg):
		allblackimg[i][j] = 0
	shadows = []
	for (i, j), num in np.ndenumerate(shadowimg):
		if shadowimg[i][j] > ShadowGroundTruth:
			oneshadowimg = GenerateOneShadow(i, j)
			shadows.append(oneshadowimg)
			#cv2.imwrite('subshadow' + str(len(shadows)) + '.png', oneshadowimg)

	'''
	img1 = cv2.pyrDown(img)
	temp_img1 = cv2.pyrDown(img1)
	temp = cv2.pyrUp(temp_img1)
	print img1
	print temp
	img2 = img1 - temp
	for i in xrange(168):
		for j in xrange(img2[i].size):
			img2[i][j] += 128
	print img2
	#cv2.imwrite('lpls.png', img2)
	'''

	cv2.imwrite(savepath + 'ori.png', img)
	for i in shadows:
		img = RemoveOneShadow(img, i)
	cv2.imwrite(savepath + resultname + '.png', img)
	#cv2.imwrite('shadowremoval.png', RemoveShadow(img, shadowimg))

if __name__ == '__main__':
	import os
	import re
	testdataset = 'UIUC'
	filename = [re.sub(r'\.jpg', '', x) for x in os.listdir('../data/UIUC/original')]
	for i in xrange(10):
		print 'do', i
		ShadowRemoval(testdataset, filename[i], './results/', str(i))