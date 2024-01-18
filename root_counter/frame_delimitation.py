import cv2
from .utils import angle_between
import copy
import numpy as np

def get_parallel_and_perpendicular_lines(image,rho = 1,theta = np.pi / 180,applythreshold = True, threshold = 15,min_line_length = 800,max_line_gap = 150):
    """_summary_

    Args:
        image (_type_): _description_
        rho (int, optional): distance resolution in pixels of the Hough grid. Defaults to 1.
        theta (_type_, optional): angular resolution in radians of the Hough grid. Defaults to np.pi/180.
        threshold (int, optional): minimum number of votes (intersections in Hough grid cell). Defaults to 15.
        min_line_length (int, optional):  minimum number of pixels making up a line. Defaults to 800.
        max_line_gap (int, optional): maximum gap in pixels between connectable line segments. Defaults to 100.

    Returns:
        _type_: _description_
    """
    img = image.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if applythreshold:
        _, thresh = cv2.threshold(imgray, 80, 255, 0)
    else:
        thresh = imgray

    lines  = cv2.HoughLinesP(thresh, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
    canvasinero = np.zeros(image.shape)

    possiblepoints = []
    for line in lines:
        for x1,y1,x2,y2 in line:

            ang = angle_between((x2,y2),(x1,y1))
            if ang >= 0 and ang < 1 or ang > 170 and ang < 190:
                
                possiblepoints.append([(x1,y1),(x2,y2)])
                imgwithlines = cv2.putText(canvasinero, '{:.2f}'.format(ang), (x1,y1),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
                cv2.line(canvasinero,(x1,y1),(x2,y2),(255,0,0),5)
            elif ang > 89 and ang < 91 or ang > 260 and ang < 290:
                
                cv2.line(canvasinero,(x1,y1),(x2,y2),(255,0,0),5)
                possiblepoints.append([(x1,y1),(x2,y2)])
                imgwithlines = cv2.putText(canvasinero, '{:.2f}'.format(ang), (x1,y1),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
    
    return imgwithlines, possiblepoints


def canvas_lines(coords, width, height, height_min = 800, width_min = 500):
    xmax = width
    xmin = 0
    ymax = height
    ymin = 0
    
    for (x1,y1),(x2,y2) in coords:

        cxmx = x1 if x1 >= x2 else x2
        cymx = y1 if y1 >= y2 else y2
        cxmn = x1 if x1 <= x2 else x2
        cymn = y1 if y1 <= y2 else y2
        ang = angle_between((x2,y2),(x1,y1))
        #vertical line
        if(cxmn < width_min and (ang > 88 and ang < 92)):
            xmin = cxmn if cxmn > xmin else xmin
        if(cxmx > (width-width_min) and (ang > 88 and ang < 92)):
            xmax = cxmx if cxmx < xmax else xmax
        
        # horizontal lines
        if(ang >= 0 and ang < 2):
            
            if(cymn < height_min):
                ymin = cymn if cymn > ymin else ymin
            
            if(cymx > (height-height_min)):
                ymax = cymx if cymx < ymax else ymax
            
    return (xmin, xmax), (ymin, ymax)    

class Frame(object):
    
    
    def frame_default(self, **kwargs):
        self.find_line_candidates(**kwargs)
        self.frame_coords()
        frameimage = self.image.copy()
        
        (xmin, xmax), (ymin, ymax) = self._frame_coords 
        cv2.line(frameimage,(xmin,ymin),(xmin,ymax),(0,0,255),5)
        cv2.line(frameimage,(xmax,ymin),(xmax,ymax),(0,0,255),5)
        cv2.line(frameimage,(xmin,ymin),(xmax,ymin),(0,0,255),5)
        cv2.line(frameimage,(xmin,ymax),(xmax,ymax),(0,0,255),5)
        
        self._frame_image = frameimage
        return (xmin, xmax), (ymin, ymax)
        
    
    def frame_coords(self, **kwargs):
        
        (xmin, xmax), (ymin, ymax) = canvas_lines(self._candidate_lines, 
                                                  width= self.image.shape[1], height=self.image.shape[0], **kwargs)
            
        self._frame_coords = (xmin, xmax), (ymin, ymax)
        
        
    def find_line_candidates(self, **kwargs):
        imglines, lines = get_parallel_and_perpendicular_lines(self.image.copy(), **kwargs)
        self._img_with_lines = imglines
        self._candidate_lines = lines
        
    def __init__(self, image) -> None:
        
        self.image = copy.deepcopy(image)