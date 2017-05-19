import cv2
import numpy as np
import copy

def RemoveShadow(originalpng, shadowpng): # get gray image
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
			res[i][j] = alpha + lamida * res[i][j]
	return res

testimgname = '022'
img = cv2.imread('../data/' + testimgname + '.jpg',0)
shadowimg = cv2.imread('../data/' + testimgname + '.png', 0)
img1 = cv2.pyrDown(img)
temp_img1 = cv2.pyrDown(img1)
temp = cv2.pyrUp(temp_img1)
print img1
print temp
img2 = img1 - temp
cv2.imwrite('ori.png', img)
for i in xrange(168):
	for j in xrange(img2[i].size):
		img2[i][j] += 128
print img2
#cv2.imwrite('lpls.png', img2)
cv2.imwrite('shadowremoval.png', RemoveShadow(img, shadowimg))