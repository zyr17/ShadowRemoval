import cv2
import numpy as np
import copy
FourNear = [[1,0],[-1,0],[0,1],[0,-1]]
ShadowGroundTruth = 128


def RemoveOneShadow(originalpng, shadowlist, illustd, illumean): # get gray image
	shadownum = []
	for [i, j] in shadowlist:
		shadownum.append(originalpng[i][j])
	shadowstd = np.std(shadownum)
	lamida = illustd / shadowstd
	alpha = illumean - lamida * np.mean(shadownum)
	for [i, j] in shadowlist:
		#temp = np.uint8(originalpng[i][j])
		temp = np.int32(originalpng[i][j])
		temp = alpha + lamida * temp
		
		if temp > 255:
			temp = 255
		if temp < 0:
			temp = 0
		
		originalpng[i][j] = np.uint8(temp)

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

def DirectShadowRemoval(testdataset, testimgname, savepath, resultname):
	img = cv2.imread('../data/' + testdataset + '/original/' + testimgname + '.jpg',0)
	global shadowimg
	shadowimg = cv2.imread('../data/' + testdataset + '/groundtruth/' + testimgname + '.png', 0)
	cv2.imwrite(savepath + 'shadow.png', shadowimg)
	global allblackimg
	allblackimg	= copy.deepcopy(shadowimg)
	for (i, j), num in np.ndenumerate(allblackimg):
		allblackimg[i][j] = 0
	illuimg = copy.deepcopy(shadowimg)
	for i in xrange(illuimg.shape[0]):
		for j in xrange(illuimg.shape[1]):
			illuimg[i][j] = 255 - illuimg[i][j]
	illunum = []
	for (i, j), num in np.ndenumerate(illuimg):
		if num > ShadowGroundTruth:
			illunum.append(img[i][j])
	illustd = np.std(illunum)
	illumean = np.mean(illunum)
	shadows = []
	for (i, j), num in np.ndenumerate(shadowimg):
		if shadowimg[i][j] > ShadowGroundTruth:
			oneshadowimg = GenerateOneShadow(i, j)
			shadows.append(oneshadowimg)
			#cv2.imwrite('subshadow' + str(len(shadows)) + '.png', oneshadowimg)
	cv2.imwrite(savepath + 'ori.png', img)
	for i in shadows:
		shadowlist = []
		for (x, y), num in np.ndenumerate(i):
			if num > ShadowGroundTruth:
				shadowlist.append([x, y])
		RemoveOneShadow(img, shadowlist, illustd, illumean)
	cv2.imwrite(savepath + resultname + '.png', img)
	#cv2.imwrite('shadowremoval.png', RemoveShadow(img, shadowimg))
	
def PyramidShadowRemoval(testdataset, testimgname, savepath, resultname, pyramidnumber):
	img = cv2.imread('../data/' + testdataset + '/original/' + testimgname + '.jpg',0)
	oriimg = copy.deepcopy(img)
	cv2.imwrite('pyramidorigin.png', img)
	global shadowimg
	shadowimg = cv2.imread('../data/' + testdataset + '/groundtruth/' + testimgname + '.png', 0)
	orishadowimg = copy.deepcopy(shadowimg)
	cv2.imwrite(savepath + 'shadow.png', shadowimg)
	global allblackimg
	allblackimg	= copy.deepcopy(shadowimg)
	allblackimg	= copy.deepcopy(shadowimg)
	for (i, j), num in np.ndenumerate(allblackimg):
		allblackimg[i][j] = 0
	illuimg = copy.deepcopy(shadowimg)
	for i in xrange(illuimg.shape[0]):
		for j in xrange(illuimg.shape[1]):
			illuimg[i][j] = 255 - illuimg[i][j]
	shadows = []
	for (i, j), num in np.ndenumerate(shadowimg):
		if shadowimg[i][j] > ShadowGroundTruth:
			oneshadowimg = GenerateOneShadow(i, j)
			shadows.append(oneshadowimg)
	pyramidpic = []
	for i in xrange(pyramidnumber):
		tempimg = cv2.pyrDown(img)
		retimg = cv2.pyrUp(tempimg)
		for j in xrange(len(img.shape)):
			retimg = np.delete(retimg, range(img.shape[j], retimg.shape[j]), j)
		retimg = img - retimg
		for x in xrange(retimg.shape[0]):
			for y in xrange(retimg.shape[1]):
				retimg[x][y] += 128
		pyramidpic.append(retimg)
		cv2.imwrite('pyramid' + str(i) + '.png', retimg)
		img = tempimg
	cv2.imwrite('pyramid' + str(pyramidnumber) + '.png', img)
	pyramidpic.append(img)
	illustds = []
	illumeans = []
	for T in xrange(len(pyramidpic)):
		illunum = []
		for (i, j), num in np.ndenumerate(illuimg):
			if num > ShadowGroundTruth:
				illunum.append(pyramidpic[T][i][j])
		illustds.append(np.std(illunum))
		illumeans.append(np.mean(illunum))
		illuimg = cv2.pyrDown(illuimg)
	
	
	for shadow in shadows:
		for i in xrange(len(pyramidpic)):		
			shadowlist = []
			for (x, y), num in np.ndenumerate(shadow):
				if num > ShadowGroundTruth:
					shadowlist.append([x, y])
			RemoveOneShadow(pyramidpic[i], shadowlist, illustds[i], illumeans[i])
			cv2.imwrite('pyramid' + str(i) + 'res.png', pyramidpic[i])
			shadow = cv2.pyrDown(shadow)
			#shadow = np.delete(shadow, np.s_[1::2], 0)
			#shadow = np.delete(shadow, np.s_[1::2], 1)
	pyramidpic.reverse()
	img = pyramidpic[0]
	for i in xrange(1, len(pyramidpic)):
		img = cv2.pyrUp(img)
		for j in xrange(len(pyramidpic[i].shape)):
			img = np.delete(img, range(pyramidpic[i].shape[j], img.shape[j]), j)
		for x in xrange(pyramidpic[i].shape[0]):
			for y in xrange(pyramidpic[i].shape[1]):
				pyramidpic[i][x][y] += 128
		img += pyramidpic[i]
	
	#replace all pixel which not in shadow by original pixel, to remove highlight around shadow
	for x in xrange(img.shape[0]):
		for y in xrange(img.shape[1]):
			if orishadowimg[x][y] <= ShadowGroundTruth:
				img[x][y] = oriimg[x][y]
	
	cv2.imwrite('pyramidres.png', img)

if __name__ == '__main__':
	import os
	import re
	testdataset = 'SBU'
	filename = [re.sub(r'\.jpg', '', x) for x in os.listdir('../data/' + testdataset + '/original')]
	#print filename
	for i in xrange(0, 10):
		print 'do', i
		#DirectShadowRemoval(testdataset, filename[i], './results/', str(i))
		PyramidShadowRemoval(testdataset, filename[i], './results/', str(i), 2)
		print 'done', i
		raw_input()