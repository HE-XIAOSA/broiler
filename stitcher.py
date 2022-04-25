import cv2 as cv
import os

def stitch_img(path):
    stitcher = cv.Stitcher_create(1)
    imgs = os.listdir(path)
    # num_img = len(imgs +'images')
    images = []
    for path in imgs:
        # print(path)
        path = "C:\\Users\\user\\Desktop\\a\\" + path
        image = cv.imread(path)
        # print(image)
        images.append(image)
    # print(images)
    result = stitcher.stitch(images)
    # cv.imshow("pinjie1",img1)
    # cv.imshow("pinjie2",img2)
    cv.namedWindow('normal_win', cv.WINDOW_NORMAL)
    cv.imshow("normal_win", result[1])
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("result.jpg", result[1])


stitch_img(path='C:\\Users\\user\\Desktop\\a')

