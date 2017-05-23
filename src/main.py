import cv2
import os
import re
import copy
import ShadowDetect
import ShadowRemoval

testdataset = 'SBU'
filename = sorted([re.sub(r'\.jpg', '', x) for x in os.listdir('../data/' + testdataset + '/original' + '/../../../src/temp')])
'''
img = cv2.imread('../data/SBU/original/lssd0.jpg')
shadow = ShadowDetect.ShadowDetect(img)
ShadowRemoval.PyramidShadowRemovalColorByImg(img, shadow, 0)
cv2.imwrite('result.jpg', img)
exit(0)
'''
for T in xrange(0, len(filename)):
	print 'do', T
	img = cv2.imread('../data/' + testdataset + '/original/' + filename[T] + '.jpg')
	shadowgroundtruth = cv2.imread('../data/' + testdataset + '/groundtruth/' + filename[T] + '.png', 0)
	#shadow = ShadowDetect.ShadowDetect(img)
	shadow = shadowgroundtruth
	print 'detect done', filename[T], T
	if shadow is None:
		print 'no shadow', filename[T], T
		continue
	img2 = copy.deepcopy(img)
	img2[shadow != 0] = (255, 255, 255)
	#cv2.imwrite('./detectresults/' + filename[T] + '.jpg', img2)
	#cv2.imwrite('./detectresults/' + filename[T] + '.png', shadow)
	ShadowRemoval.PyramidShadowRemovalColorByImg(img, shadow, 0)
	print 'remove done', filename[T], T
	cv2.imwrite('./results/' + filename[T] + '.jpg', img)