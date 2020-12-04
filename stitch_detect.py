import cv2
import imutils

# Reading the Images
left = cv2.imread("left.png")
right = cv2.imread("right.png")

cv2.imshow("Image_1", left)
cv2.imshow("Image_2", right)

images = []
images.append(left)
images.append(right)

#Stitching the two images
stitcher = cv2.Stitcher.create()
ret, pano = stitcher.stitch(images)

#It will only stitch if the left and right image has something common
if ret==cv2.STITCHER_OK:
    cv2.imshow("Stitched_image", pano)

    # Initializing the HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detecting all the regions in the image that has a pedestrians inside it
    (regions, _) = hog.detectMultiScale(pano,
                                        winStride=(4, 4),
                                        padding=(4, 4),
                                        scale=1.05)

    # Drawing the regions in the Image
    for (x, y, w, h) in regions:
        cv2.rectangle(pano, (x, y),
                      (x + w, y + h),
                      (0, 0, 255), 2)

    # Showing the output Image
    cv2.imshow("Final_output", pano)
    cv2.waitKey()

else:
    print("Not possible")

cv2.destroyAllWindows()
