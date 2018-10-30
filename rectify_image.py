#coding=utf-8
import numpy as np
import cv2

from skimage.filters import try_all_threshold
import matplotlib.pyplot as plt
from skimage.filters import threshold_mean




target_path = 'samples/bandcard_20180802110747_06e0d7ec8a4bcf3579325ed8a92a5743.jpg'
# target_path = 'samples/bandcard_20180802111107_51c414f8292969119d01d826ebcf8ecf.jpg'
target_path = 'samples/bandcard_20180827124218_d282e1e29118ebc37654d20dbac8e43f.jpg'
target_path = 'samples/bandcard_20180802110756_231750e6826e433413b941523ad6088d.jpg'
target_path = 'samples/bandcard_20180802110916_c36c19b4e9b492632053b2e222d7f9a6.jpg'
target_path='samples/bandcard_20180822144013_981382785c6a4308636ccc0e464b9ea9.jpg'
target_path='samples/bandcard_20180802111004_6f5ac60f4193fe144ddd76b36881afab.jpg'
target_path='samples/bandcard_20180802110825_de7920d8e7e7a906f0abe448a9d51637.jpg'
target_path='samples/bandcard_20180802110830_782146b47eae8f0c650a22bf39dac0fc.jpg'
target_path='samples/bandcard_20180802110856_7ae1659b49633101db537f9b67aff94e.jpg'
target_path='samples/bandcard_20180827124250_86f7b08c3f334aad782b3f8ded636c39.jpg'

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), cv2.INTER_LINEAR)

    # return the warped image
    return warped


imgnp = cv2.imread(target_path)
imgnp_copy = imgnp.copy()
cv2.imwrite('test_dir/origin.png', imgnp)


# 边界轮廓检测
# 高斯模糊，模糊银行卡内的边界
gaussianResult = cv2.GaussianBlur(imgnp_copy, (5,5), 0)
cv2.imwrite('test_dir/gaussianResult.png', gaussianResult)
gray_imgnp = cv2.cvtColor(gaussianResult, cv2.COLOR_BGR2GRAY) # gray
cv2.imwrite('test_dir/gray.png', gray_imgnp)
h, w = gray_imgnp.shape
# binary
from skimage.filters import threshold_minimum



thresh_min = threshold_minimum(gray_imgnp)
binary_imgnp = ((gray_imgnp < thresh_min)*255).astype(np.uint8)
cv2.imwrite('test_dir/binary_test.png', binary_imgnp)

# fig, ax = try_all_threshold(gray_imgnp, figsize=(10, 8), verbose=False)
# plt.show()


# warped = threshold_local(warped_gray, block_size, offset=10)
# warped_adaptive = (warped_gray > warped).astype("uint8")*255
# ret1, binary_imgnp = cv2.threshold(gray_imgnp, 127, 255, cv2.THRESH_BINARY_INV)
# binary_imgnp = cv2.adaptiveThreshold(gray_imgnp,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY,11,2 )
# cv2.imwrite('test_dir/binary.png', binary_imgnp)



# 膨胀
kernel = np.ones((20, 20), np.uint8)
dilated = cv2.dilate(binary_imgnp, kernel)
cv2.imwrite('test_dir/dilated.png', dilated)
# 腐蚀
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(dilated, kernel)
cv2.imwrite('test_dir/eroded.png', eroded)




# 边界检测
# im2, contours, hierarchy = cv2.findContours(binary_imgnp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

edges = auto_canny(eroded)

print("edges:", np.mean(edges), np.max(edges), np.min(edges), np.unique(edges))

cv2.imwrite('test_dir/edges.png', edges)

# 把边缘检测结果转换为灰度图
# bgr_canny = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # gray
# cv2.imwrite('test_dir/bgr_canny.png', bgr_canny)



# 检测边
lines = cv2.HoughLinesP(edges, rho=0.1, theta=np.pi/180, threshold=10, minLineLength=15, maxLineGap=20)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(imgnp_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite("line_detect_possible_demo.jpg", imgnp_copy)



# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.RETR_FLOODFILL)
cnts = cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse = True)[:10]

cv2.drawContours(imgnp_copy, cnts, -1, (0,255,0), 3)
cv2.imwrite('test_dir/contours.png', imgnp_copy)

print("len cnts:", len(cnts))
# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05* peri, True)
    #print "approx:", approx

    # 找最大的外包矩形


    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        print '111111'
        screenCnt = approx
        break

copy = imgnp.copy()
cv2.drawContours(copy, [screenCnt], -1, (0, 255, 0), 2)
cv2.imwrite('test_dir/countours.png', copy)

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(imgnp, screenCnt.reshape(4, 2) )
cv2.imwrite('test_dir/wrapped.png', warped)

# # convert the warped image to grayscale, then threshold it
# # to give it that 'black and white' paper effect
# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# T = threshold_local(warped, 11, offset=10, method="gaussian")
# warped = (warped > T).astype("uint8") * 255
#
# # show the original and scanned images
# print("STEP 3: Apply perspective transform")
# cv2.imshow("Original", imutils.resize(orig, height=650))
# cv2.imshow("Scanned", imutils.resize(warped, height=650))
# cv2.waitKey(0)



# minLineLength = 50
# maxLineGap = 50
# lines = cv2.HoughLines(gray_imgnp.copy(),1,np.pi/180,200)
# print lines
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(imgnp,(x1,y1),(x2,y2),(0,255,0),2)
# cv2.imwrite('test_dir/line.png', imgnp)
# cv2.drawContours(imgnp, contours, -1, (0,255,0), 3)
# cv2.imwrite('test_dir/contours.png', imgnp)
