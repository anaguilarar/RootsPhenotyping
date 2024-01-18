import numpy as np
import cv2
from skimage.morphology import skeletonize, thin
from skimage.filters import meijering

def angle_between(p1, p2):
    
    ang = np.arctan((p2[1] - p1[1])/(p2[0] - p1[0]))
    
    return abs(np.rad2deg(ang))


def get_contours(image):
    
    if np.max(image)<150:
        image = image*255
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours


def image_counturs(image, minarea = 3000, maxarea = 90000):
    
    contours = get_contours(image)
    newcont = []
    allcntareas = []
    for cnt in contours:
        rect = cv2.contourArea(cnt)
        allcntareas.append(rect)
        if rect >  minarea and  rect <  maxarea:
            newcont.append(cnt)
            
    return tuple(newcont)


def fill_contours(contours,imgshape):
    canvasinzero = np.zeros(imgshape)
    for cnt in contours:
        cv2.fillPoly(canvasinzero, pts=[cnt], color=(255,0,0))

    canvasinzero[canvasinzero<0] = 0
    canvasinzero[canvasinzero>255] = 0

    return canvasinzero.astype(np.uint8)

def image_preprocessing(image, togray = True, apply_morphological = False,bilateralfilter=False,filter2d = False,
                        lines_preprocess = True, dilation = True, 
                        skeleton = True,
                        meijiring_filter = True, min_thresh = 60,
                        remove_center = False, thval= 0.80, bhval= 0.20):
    
    if bilateralfilter:
        img = cv2.bilateralFilter(image.copy(),9,75,75)
    else:
        img =image.copy()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if togray else img
        
    if apply_morphological:
        imgray = cv2.morphologyEx(imgray, cv2.MORPH_OPEN, kernel)
        
    ## filter
    
    
    
    canvasinzero = np.zeros(image.shape)

    if min_thresh is not None:
        ret, thresh = cv2.threshold(imgray, min_thresh, 255, 0)
    else:
        thresh = imgray
        
    
    if filter2d :
        linekernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        thresh = cv2.filter2D(thresh,-1,linekernel)
    if remove_center and min_thresh is not None:
        hh, ww = image.shape[:2]
        thresh[int(hh*bhval):int(hh*thval),int(ww*0.10):int(ww*0.90)] = 0    
        
    if lines_preprocess:
        lines  = cv2.HoughLinesP(thresh, 1, np.pi / 180, 15 )
        
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(canvasinzero,(x1,y1),(x2,y2),(255,0,0),5)
        canvasinzero = canvasinzero[:,:,0]
    else:
        canvasinzero = thresh
    result = cv2.dilate(canvasinzero, kernel)
    result = cv2.erode(result, (5, 5), iterations = 1)

    if skeleton:
        result = skeletonize(result/255)
        result = ((result)*255).astype(np.uint8)
    if dilation:
        result = cv2.dilate(result, kernel)
    if meijiring_filter:
        result = meijering(result, black_ridges=True,sigmas= range(3, 4))
        if remove_center and min_thresh is None:
            hh, ww = image.shape[:2]
            result[int(hh*bhval):int(hh*thval),int(ww*0.10):int(ww*0.90)] = 0 
        
    areathresholds = [5000,  90000]
    if lines_preprocess:
        areathresholds = [4000,  90000]
    else:
        areathresholds = [5000,  90000]

    if not meijiring_filter:
        areathresholds = [500,  90000]

    return result, areathresholds
