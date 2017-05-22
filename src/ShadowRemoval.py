import cv2
import numpy as np
import copy
FourNear = [[1,0],[-1,0],[0,1],[0,-1]]
EightNear = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]]
ShadowGroundTruth = 128


def RemoveShadowStrip(originalpng, shadowlist, illustd, illumean): # get gray image
	if shadowlist == None or len(shadowlist) == 0:
		return
	shadownum = []
	for [i, j] in shadowlist:
		shadownum.append(originalpng[i][j])
	shadowstd = np.std(shadownum)
	if shadowstd == 0:
		return
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

def RemoveOneShadow(originalpng, shadowlist, illustd, illumean, thickness = 10000):
	#if thickness == 10000:
	#	RemoveShadowStrip(originalpng, shadowlist, illustd, illumean)
	#	return
	n = originalpng.shape[0]
	m = originalpng.shape[1]
	shadow = originalpng.astype(int)
	for i in xrange(n):
		for j in xrange(m):
			shadow[i][j] = - 1
	for [i, j] in shadowlist:
		shadow[i][j] = 0
	striplist = [[], []]
	for [i, j] in shadowlist:
		flag = False
		for [x, y] in FourNear:
			x += i
			y += j
			if x >= 0 and y >= 0 and x < n and y < m:
				if shadow[x][y] == - 1:
					flag = True
		if flag:
			shadow[i][j] = 1
			striplist[1].append([i, j])
	nowstrip = 1
	while len(striplist[nowstrip]) > 0:
		templist = []
		for [i, j] in striplist[nowstrip]:
			for [x, y] in FourNear:
				x += i
				y += j
				if x >= 0 and y >= 0 and x < n and y < m:
					if shadow[x][y] == 0:
						shadow[x][y] = nowstrip + 1
						templist.append([x, y])
		striplist.append(templist)
		nowstrip += 1
	nowstrip = 0
	if thickness > 1:
		while nowstrip < len(striplist):
			for i in xrange(nowstrip + 1, min(len(striplist), nowstrip + thickness)):
				striplist[nowstrip] += striplist[i]
			RemoveShadowStrip(originalpng, striplist[nowstrip], illustd, illumean)
			nowstrip += thickness
	else:
		MINLENGTH = 200
		SEGMENTS = 3
		for nowstrip in striplist:
			maxlength = max(MINLENGTH, len(nowstrip) / SEGMENTS)
			for p in nowstrip:
				if shadow[p[0]][p[1]] == 0:
					continue
				stripnum = shadow[p[0]][p[1]]
				shadow[p[0]][p[1]] = 0
				tmplist = [p]
				t = 0
				while t < len(tmplist):
					for x, y in EightNear:
						x += tmplist[t][0]
						y += tmplist[t][1]
						if x >= 0 and y >= 0 and x < n and y < m:
							if shadow[x][y] == stripnum:
								shadow[x][y] = 0
								tmplist.append([x, y])
								if len(tmplist) >= maxlength:
									break
					if len(tmplist) >= maxlength:
						break
					t += 1
				RemoveShadowStrip(originalpng, tmplist, illustd, illumean)

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
			oneshadowimg[x][y] = shadowimg[x][y]
			shadowimg[x][y] = 0
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
	oneilluimg = copy.deepcopy(allblackimg)
	
	MinBorder = 5
	MaxBorder = 10
	for (i, j), num in np.ndenumerate(oneshadowimg):
		if num > ShadowGroundTruth:
			for x, y in FourNear:
				X = x * MaxBorder + i
				Y = y * MaxBorder + j
				if X < 0 or Y < 0 or X >= shadowimg.shape[0] or Y >= shadowimg.shape[1]:
					continue
				if oneshadowimg[X][Y] > ShadowGroundTruth:
					continue
				if oneshadowimg[X - x * (MaxBorder - MinBorder)][Y - y * (MaxBorder - MinBorder)] > ShadowGroundTruth:
					continue
				if shadowimg[X][Y] > ShadowGroundTruth:
					continue
				if shadowimg[X - x * (MaxBorder - MinBorder)][Y - y * (MaxBorder - MinBorder)] > ShadowGroundTruth:
					continue
				oneilluimg[X][Y] = 255
	OutBorder = 5
	changelist = []
	for (i, j), num in np.ndenumerate(oneshadowimg):
		if num > ShadowGroundTruth:
			for x, y in FourNear:
				X = x * OutBorder + i
				Y = y * OutBorder + j
				if X < 0 or Y < 0 or X >= shadowimg.shape[0] or Y >= shadowimg.shape[1]:
					continue
				changelist.append([X, Y])
	for i, j in changelist:
		oneshadowimg[i][j] = 255
	return oneshadowimg, oneilluimg
	
def PyramidShadowRemovalBlack(img, inputshadowimg, pyramidnumber):
	oriimg = copy.deepcopy(img)
	cv2.imwrite('pyramidorigin.png', img)
	global shadowimg
	shadowimg = inputshadowimg
	orishadowimg = copy.deepcopy(shadowimg)
	cv2.imwrite('shadow.png', shadowimg)
	global allblackimg
	allblackimg	= copy.deepcopy(shadowimg)
	for (i, j), num in np.ndenumerate(allblackimg):
		allblackimg[i][j] = 0
	illuimg = copy.deepcopy(shadowimg)
	for i in xrange(illuimg.shape[0]):
		for j in xrange(illuimg.shape[1]):
			illuimg[i][j] = 255 - illuimg[i][j]
	shadows = []
	illus = []
	for (i, j), num in np.ndenumerate(shadowimg):
		if shadowimg[i][j] > ShadowGroundTruth:
			oneshadowimg, oneilluimg = GenerateOneShadow(i, j)
			shadows.append(oneshadowimg)
			illus.append(oneilluimg)
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
	
	for shadownum, shadow in enumerate(shadows):
		illu = illus[shadownum]
		for i in xrange(len(pyramidpic)):
			shadowlist = []
			for (x, y), num in np.ndenumerate(shadow):
				if num > ShadowGroundTruth:
					shadowlist.append([x, y])
			
			illunum = []
			for (x, y), num in np.ndenumerate(illu):
				if num > ShadowGroundTruth:
					illunum.append(pyramidpic[i][x][y])
			illustd = np.std(illunum)
			illumean = np.mean(illunum)
			illu = cv2.pyrDown(illu)
			
			if i == len(pyramidpic) - 1:
				RemoveOneShadow(pyramidpic[i], shadowlist, illustd, illumean, 1)
			else:
				RemoveOneShadow(pyramidpic[i], shadowlist, illustd, illumean)	
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
	'''
	#replace all pixel which not in shadow by original pixel, to remove highlight around shadow
	for x in xrange(img.shape[0]):
		for y in xrange(img.shape[1]):
			if orishadowimg[x][y] <= ShadowGroundTruth:
				img[x][y] = oriimg[x][y]
	'''
	#cv2.imwrite('imgres.png', img)
	#print 'show img'
	return img

def PyramidShadowRemovalColor(testdataset, testimgname, resultname, pyramidnumber):
	#colorimg = cv2.imread('../data/' + testdataset + '/original/' + testimgname + '.jpg', 0)
	inputshadowimg = cv2.imread('../data/' + testdataset + '/groundtruth/' + testimgname + '.png', 0)
	
	colorimg = cv2.imread('../data/' + testdataset + '/original/' + testimgname + '.jpg')
	color0img = np.delete(colorimg, [1, 2], 2).reshape((colorimg.shape[0], colorimg.shape[1]))
	color1img = np.delete(colorimg, [0, 2], 2).reshape((colorimg.shape[0], colorimg.shape[1]))
	color2img = np.delete(colorimg, [0, 1], 2).reshape((colorimg.shape[0], colorimg.shape[1]))
	color0img = PyramidShadowRemovalBlack(color0img, copy.deepcopy(inputshadowimg), pyramidnumber)
	#raw_input()
	color1img = PyramidShadowRemovalBlack(color1img, copy.deepcopy(inputshadowimg), pyramidnumber)
	#raw_input()
	color2img = PyramidShadowRemovalBlack(color2img, copy.deepcopy(inputshadowimg), pyramidnumber)
	#raw_input()
	for i in xrange(colorimg.shape[0]):
		for j in xrange(colorimg.shape[1]):
			colorimg[i][j] = [color0img[i][j], color1img[i][j], color2img[i][j]]
	cv2.imwrite(resultname + ' - ' + testimgname + '.png', colorimg)
	'''
	PyramidShadowRemovalBlack(colorimg, copy.deepcopy(inputshadowimg), pyramidnumber)
	cv2.imwrite(resultname + ' - ' + testimgname + '.png', colorimg)
	'''

def PyramidShadowRemovalColorByImg(colorimg, inputshadowimg, testimgname, resultname, pyramidnumber):
	color0img = np.delete(colorimg, [1, 2], 2).reshape((colorimg.shape[0], colorimg.shape[1]))
	color1img = np.delete(colorimg, [0, 2], 2).reshape((colorimg.shape[0], colorimg.shape[1]))
	color2img = np.delete(colorimg, [0, 1], 2).reshape((colorimg.shape[0], colorimg.shape[1]))
	color0img = PyramidShadowRemovalBlack(color0img, copy.deepcopy(inputshadowimg), pyramidnumber)
	color1img = PyramidShadowRemovalBlack(color1img, copy.deepcopy(inputshadowimg), pyramidnumber)
	color2img = PyramidShadowRemovalBlack(color2img, copy.deepcopy(inputshadowimg), pyramidnumber)
	for i in xrange(colorimg.shape[0]):
		for j in xrange(colorimg.shape[1]):
			colorimg[i][j] = [color0img[i][j], color1img[i][j], color2img[i][j]]
	cv2.imwrite(resultname + ' - ' + testimgname + '.png', colorimg)
	'''
	PyramidShadowRemovalBlack(colorimg, copy.deepcopy(inputshadowimg), pyramidnumber)
	cv2.imwrite(resultname + ' - ' + testimgname + '.png', colorimg)
	'''

if __name__ == '__main__':
	import os
	import re
	testdataset = 'SBU'
	filename = [re.sub(r'\.jpg', '', x) for x in os.listdir('../data/' + testdataset + '/original')]
	#print filename
	for i in xrange(0, 10):
		print 'do', i
		PyramidShadowRemovalColor(testdataset, filename[i], './results/' + str(i), 0)
		#PyramidShadowRemovalColor(testdataset, filename[i], './' + str(i), 2)
		print 'done', i