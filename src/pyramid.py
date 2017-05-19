import cv2

def RemoveShadow(originalpng, shadowpng): # get gray image
	

img = cv2.imread('../data/test.jpg',0) #
img1 = cv2.pyrDown(img)#
temp_img1 = cv2.pyrDown(img1)
temp = cv2.pyrUp(temp_img1)
print img1
print temp
img2 = img1 - temp #
cv2.imwrite('gauss.png', img1)
for i in xrange(210):
	for j in xrange(img2[i].size):
		img2[i][j] += 128
print img2
cv2.imwrite('lpls.png', img2)