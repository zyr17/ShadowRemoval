import copy
import cv2
import numpy as np

def LogConvert(img):
	res = np.array([[0.0 for y in xrange(img.shape[1])] for x in xrange(img.shape[0])])
	for i in xrange(res.shape[0]):
		for j in xrange(res.shape[1]):
			res[i][j] = (1. + max(img[i][j])) / (1. + min(img[i][j]))
	res = np.log2(res)
	res += min(res.flat)
	temp = 255 / max(res.flat)
	res *= temp
	return res

def WaterShed(inputimg):
	
	gray = copy.deepcopy(inputimg).astype('uint8')
	img = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
	
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	
	# noise removal
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

	# sure background area
	sure_bg = cv2.dilate(opening,kernel,iterations=3)

	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
	ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)
	
	# Marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)

	# Add one to all labels so that sure background is not 0, but 1
	markers = markers+1

	# Now, mark the region of unknown with zero
	markers[unknown==255] = 0
	
	markers = cv2.watershed(img,markers)
	#m2 = copy.deepcopy(markers)
	#m2[m2 == -1] = 255
	#cv2.imwrite('water.png', m2)
	img[markers == -1] = [255,0,0]
	return img, markers

def BlockShadowDetect(img):
	M = 1.31
	N = 1.19
	K1 = 0.8
	K2 = 1.2
	'''
	logres = logres = LogConvert(img)
	cv2.imwrite('conv.png', logres)
	wsres, marker = WaterShed(logres)
	temp = copy.deepcopy(img)
	temp[marker == -1] = [255, 0, 0]
	cv2.imwrite('water.png', temp)
	'''
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	res = np.zeros((img.shape[0], img.shape[1]), np.uint8)
	marker = np.zeros((img.shape[0], img.shape[1]), np.uint8)
	nowmarkernum = 0
	resmeans = [None]
	resnsmeans = [None]
	#res = copy.deepcopy(img)
	
	STDLINE = 0.25
	ROW = COLUMN = 30
	ABS = 40
	i = j = 0
	while i < img.shape[0]:
		if i + ROW >= img.shape[0]:
			break
		while j < img.shape[1]:
			if j + COLUMN >= img.shape[1]:
				break
			now = gray[i:i + ROW, j:j + COLUMN].astype('float32')
			ker = np.ones((5, 5))
			nowres = cv2.filter2D(now, -1, ker) / 25
			#print i, j, np.std(nowres), STDLINE * (max(nowres.flat) - min(nowres.flat))
			if np.std(nowres) > STDLINE * (max(nowres.flat) - min(nowres.flat)) and max(nowres.flat) - min(nowres.flat) > ABS:
				now = marker[i:i + ROW, j:j + COLUMN]
				nowmarkernum += 1
				now[:][:] = nowmarkernum
			
			j += COLUMN
		j = 0
		i += ROW
	
	now = 1
	while now in marker:
		nowlist = img[marker == now]
		colormean = np.mean(nowlist, axis = 0)
		tmparray = np.array([x for x in nowlist if x[0] > colormean[0] and x[1] > colormean[1] and x[2] > colormean[2]])
		nsmean = np.mean(tmparray, axis = 0)
		if tmparray is None or len(tmparray.flat) == 0:
			now += 1
			resmeans.append(None)
			resnsmeans.append(None)
			continue
		L = nsmean * [1, N ,M]
		L /= L[0]
		maxnum = 0
		minnum = 0
		for i in xrange(3):
			if L[i] > L[maxnum]:
				maxnum = i
			if L[i] < L[minnum]:
				minnum = i
		
		X = np.array([i[maxnum].astype('int32') - i[minnum].astype('int32') for i in nowlist])
		T = np.mean(X)
		#print maxnum, minnum, X, T
		shadowlist = np.array([nowlist[i] for i in xrange(len(X)) if X[i] < T and nowlist[i] not in tmparray])
		if shadowlist is None or len(shadowlist.flat) == 0:
			now += 1
			resmeans.append(None)
			resnsmeans.append(None)
			continue
		shadowmean = np.mean(shadowlist, axis = 0)
		tmplist = (nsmean - shadowmean) / (nsmean[0] - shadowmean[0]) / L
		#print nsmean, shadowmean, tmplist
		if min(tmplist) >= K1 and max(tmplist) <= K2 and nsmean[0] > shadowmean[0] and nsmean[1] > shadowmean[1] and nsmean[2] > shadowmean[2]:
			resmeans.append(shadowmean)
			resnsmeans.append(nsmean)
			for i in xrange(img.shape[0]):
				for j in xrange(img.shape[1]):
					num = img[i][j]
					if marker[i][j] == now:
						if num[maxnum].astype('int32') - num[minnum] < T:
							res[i][j] = now
		else:
			resmeans.append(None)
			resnsmeans.append(None)
		now += 1
	return res, resmeans, resnsmeans

def DCmpAll3(a, b):
	if a[0] > b[0] and a[1] > b[1] and a[2] > b[2]:
		return 1
	if a[0] < b[0] and a[1] < b[1] and a[2] < b[2]:
		return -1
	return 0

def DcmpMean(a, b):
	aa = np.mean(a)
	bb = np.mean(b)
	if aa > bb:
		return 1
	if aa < bb:
		return -1
	return 0

def FloodFill(img, mask, seedpoint, newval, threshold):
	l = [seedpoint]
	t = 0
	n = img.shape[0]
	m = img.shape[1]
	FourNear = [[0,1],[0,-1],[1,0],[-1,0]]
	res = np.zeros((n, m), np.uint8)
	while t < len(l):
		i = l[t][0]
		j = l[t][1]
		for x, y in FourNear:
			x += i
			y += j
			if x < 0 or x >= n or y < 0 or y >= m:
				continue
			#print img[x][y], np.mean(img[x][y]), threshold
			if DCmpAll3(img[x][y], threshold) != -1:
				continue
			if res[x][y] != 0:
				continue
			res[x][y] = 255
			l.append([x, y])
		t += 1
	mask[res != 0] = newval
	return res

def CatInside(img):
	n = img.shape[0]
	m = img.shape[1]
	l = []
	SHADOW = 255
	TEMPCOLOR = 128
	NOTSHADOW = 0
	FourNear = [[0,1],[0,-1],[1,0],[-1,0]]
	for i in xrange(n):
		if img[i][0] == NOTSHADOW:
			l.append([i, 0])
			img[i][0] = TEMPCOLOR
		if img[i][m - 1] == NOTSHADOW:
			l.append([i, m - 1])
			img[i][m - 1] = TEMPCOLOR
	for j in xrange(m):
		if img[0][j] == NOTSHADOW:
			l.append([0, j])
			img[0][j] = TEMPCOLOR
		if img[n - 1][j] == NOTSHADOW:
			l.append([n - 1, j])
			img[n - 1][j] = TEMPCOLOR
	t = 0
	while t < len(l):
		for x, y in FourNear:
			x += l[t][0]
			y += l[t][1]
			if x < 0 or x >= n or y < 0 or y >= m:
				continue
			if img[x][y] == NOTSHADOW:
				img[x][y] = TEMPCOLOR
				l.append([x, y])
		t += 1
	img[img == NOTSHADOW] = SHADOW
	img[img == TEMPCOLOR] = NOTSHADOW

def RemoveSmall(img):
	#TODO remove small pieces of shadow and non-shadow places
	THRESHOLD = 1000
	SHADOW = 255
	NOTSHADOW = 0
	FourNear = [[0,1],[0,-1],[1,0],[-1,0]]
	done = np.zeros(img.shape, np.int32)
	n = img.shape[0]
	m = img.shape[1]
	now = 1
	for i in xrange(n):
		for j in xrange(m):
			if done[i][j] == 0 and img[i][j] == SHADOW:
				l = [[i, j]]
				done[i][j] = now
				t = 0
				while t < len(l):
					for x, y in FourNear:
						x += l[t][0]
						y += l[t][1]
						if x < 0 or x >= n or y < 0 or y >= m:
							continue
						if img[x][y] != img[i][j]:
							continue
						if done[x][y] != 0:
							continue
						l.append([x, y])
						done[x][y] = now
					t += 1
				if len(l) < THRESHOLD:
					for x, y in l:
						img[x][y] = NOTSHADOW
			now += 1
	done[done != 0] = 0
	now = 1
	for i in xrange(n):
		for j in xrange(m):
			if done[i][j] == 0 and img[i][j] == NOTSHADOW:
				l = [[i, j]]
				done[i][j] = now
				t = 0
				while t < len(l):
					for x, y in FourNear:
						x += l[t][0]
						y += l[t][1]
						if x < 0 or x >= n or y < 0 or y >= m:
							continue
						if img[x][y] != img[i][j]:
							continue
						if done[x][y] != 0:
							continue
						l.append([x, y])
						done[x][y] = now
					t += 1
				if len(l) < THRESHOLD:
					for x, y in l:
						img[x][y] = SHADOW
			now += 1

def ShadowDetect(img):
	shadowres, shadowmeans, nsmeans = BlockShadowDetect(img)
	tmpmeans = np.mean(np.array([x for x in shadowmeans if x is not None]))
	if tmpmeans is None:
		return None
	#print tmpmeans
	for j in xrange(len(shadowmeans)):
		if shadowmeans[j] is not None and np.mean(shadowmeans[j]) > tmpmeans:
			#print j
			shadowres[shadowres == j] = 255
	floodfillmask = np.zeros((img.shape[0] + 2) * (img.shape[1] + 2)).reshape((img.shape[0] + 2, img.shape[1] + 2)).astype('uint8')
	ONEDIFF = 15
	FLOODDIFF = (ONEDIFF, ONEDIFF, ONEDIFF)
	nsmean = np.mean(np.array([x for x in nsmeans if x is not None]), axis = 0)
	shadowmean = np.mean(np.array([x for x in shadowmeans if x is not None]), axis = 0)
	#print nsmean, shadowmean
	floodpos = [[0, 0, 0, (nsmean - shadowmean) * 0.5 + shadowmean] for x in xrange(len(shadowmeans))]
	for i in xrange(img.shape[0]):
		for j in xrange(img.shape[1]):
			if shadowres[i][j] != 0 and shadowres[i][j] != 255 and (shadowmeans[shadowres[i][j]] is not None):
				floodpos[shadowres[i][j]][0] += i
				floodpos[shadowres[i][j]][1] += j
				floodpos[shadowres[i][j]][2] += 1
	shadowres[:][:] = 0
	for i in floodpos:
		if i[2] != 0:
			x = i[0] / i[2]
			y = i[1] / i[2]
			if shadowres[x][y] == 255:
				continue
			FloodFill(img, shadowres, (x, y), 255, i[3])
			#print x, y#, img[x][y]
			#cv2.imwrite('shadowres.png', img)
			#raw_input()
	#img[shadowres != 0] = [255, 255, 255]
	#img[shadowres == 255] = [255, 0, 0]
	RemoveSmall(shadowres)
	#CatInside(shadowres)
	return shadowres

if __name__ == '__main__':
	import os
	import re
	testdataset = 'NAIVE'
	filename = [re.sub(r'\.jpg', '', x) for x in os.listdir('../data/' + testdataset + '/original')]
	#print filename
	for T in xrange(0, len(filename)):
		img = cv2.imread('../data/' + testdataset + '/original/' + filename[T] + '.jpg')
		cv2.imwrite('ori.png', img)
		shadowres = ShadowDetect(img)
		if shadowres is None:
			print 'not find shadow', T
			continue
		img[shadowres == 255] = (255, 255, 255)
		cv2.imwrite('./detectresults/' + str(T) + ' - ' + filename[T] + '.png', img)
		print 'done', T
		#raw_input()