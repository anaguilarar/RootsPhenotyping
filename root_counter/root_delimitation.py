import cv2
import copy
import numpy as np
from root_counter.plt_functions.utils import random_color_generator
from .frame_delimitation import Frame
from .utils import image_preprocessing, get_contours

from .individual_root import RootDrawing
from .utils import fill_contours, image_counturs,image_preprocessing
import os

import warnings


class RootImage(Frame, RootDrawing):

    def _preprocess_plate_image(self):
        meijering_filter = True
        hh, ww = self.root_image.shape[:2]
        deletecenterimage = self.root_image.copy()
        if self._direction == 'top':
                thval, bhval = 0.80, 0.05
        else:
                thval, bhval = 0.85, 0.07
                
        img,_ = image_preprocessing(self.root_image,apply_morphological=True,bilateralfilter=True,filter2d=False,
                                        meijiring_filter=meijering_filter,
                                        lines_preprocess=False,skeleton=False, min_thresh=None, 
                                        dilation=False,remove_center=True, 
                                        thval=thval, bhval=bhval)

        if meijering_filter:
                #newimg = newimg*255
            newimg = np.stack([img*255., np.zeros(img.shape),
                               np.zeros(img.shape)],axis =2).astype(np.uint8)
        else:
            newimg = np.stack([img, np.zeros(img.shape),
                               np.zeros(img.shape)],axis =2).astype(np.uint8)

        return newimg

    def estimate_all_root_lengths(self):

        assert self.individual_root_images is not None
        self.individual_main_roots  = {}
        for i in range(len(self.individual_root_images)):
            
            try:
                img = self.individual_root_images['root_{}'.format(i+1)].copy()
                mainrootimg, coords = self.detect_main_root(image=img, max_lateralrootdist = 300)
                self.individual_main_roots['root_{}'.format(i+1)] = [mainrootimg, coords]
            except:
                warnings.warn(f'it was not possible to process the root {str(i)}')
                img = self.individual_root_images['root_{}'.format(i+1)].copy()
                self.individual_main_roots['root_{}'.format(i+1)] = [np.zeros(img.shape[:2]), None]
                    

    def plot_roots_over_image(self):
    
        hh, ww = self.plate_image().shape[:2]
        imageinzeros_all =  np.zeros((hh,ww,3), dtype=np.uint8)
        coordstext = {}
        if self.individual_main_roots is None:
            self.estimate_all_root_lengths()
            
        for rootid in list(self.individual_main_roots.keys()):
            imageinzeros =  np.zeros((hh,ww,3), dtype=np.uint8)
            
            root_img = self.individual_root_images[rootid]
            cntr = self.individual_root_countours[rootid]
            x,y,w,h = cv2.boundingRect(cntr)

            npones, coords = self.individual_main_roots[rootid]
            if coords is None:
                continue
            #plt.imshow(npones)
            
            colorrgb = random_color_generator()
            image_srrot_rgb  = []
            for i in colorrgb:
                image_srrot_rgb.append(npones * i)
                
            imgrootcolor = np.stack(image_srrot_rgb, axis = 2)
            ## central coords
            heightpos = np.array([int(z.split('-')[0]) for z in coords['main']]).mean()
            widthpos = np.array([int(z.split('-')[1]) for z in coords['main']]).mean()

        
            imageinzeros[y:(y+h),x:(x+w)] = imgrootcolor
            imageinzeros_all = imageinzeros+imageinzeros_all
            coordstext[rootid] = [int(y+(heightpos)),
                                                  int(x+(widthpos))]
        #coordstext.append()

        
        imgd = imageinzeros_all.astype(np.uint8).copy()
        kernel = np.ones((15,15),np.uint8)
        imgd = cv2.dilate(imgd,kernel,iterations = 1)
        dst = cv2.addWeighted(self.plate_image(), 0.7, imgd, 0.5, 0)

        for idroot in coordstext.keys():

            (y, x) = coordstext[idroot]
            dst = cv2.putText(dst, '{}'.format(idroot), [x,y],
                                cv2.FONT_HERSHEY_PLAIN,4,(0,255,0),2,cv2.LINE_AA)
        
        return dst
        

    def plot_root_images(self):
    
        hh, ww = self.plate_image().shape[:2]
        imageinzeros_all =  np.zeros((hh,ww,3), dtype=np.uint8)
        for idroot, rootid in enumerate(list(self.individual_root_images.keys())):
            imageinzeros =  np.zeros((hh,ww,3), dtype=np.uint8)
            
            root_img = self.individual_root_images[rootid]
            cntr = self.individual_root_countours[rootid]
            npones = self.preprocess_single_root(root_img)
            #plt.imshow(npones)
            x,y,w,h = cv2.boundingRect(cntr)
            colorrgb = random_color_generator()
            image_srrot_rgb  = []
            for i in colorrgb:
                image_srrot_rgb.append(npones * i)
                
            imgrootcolor = np.stack(image_srrot_rgb, axis = 2)
            imgrootcolor = cv2.putText(imgrootcolor, 'root: {}'.format(idroot+1), (10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            
            imageinzeros[y:(y+h),x:(x+w)] = imgrootcolor
            imageinzeros_all = imageinzeros+imageinzeros_all
            
        return imageinzeros_all
        
    def estimate_root_length(self, root_id):

        img = self.individual_root_images['root_{}'.format(root_id)].copy()
        mainrootimg, coords = self.detect_main_root(image=img, max_lateralrootdist = 300)
        
        return mainrootimg, coords

    def find_all_lengths(self):
        
        if self.individual_main_roots is None:
            self.estimate_all_root_lengths()
            
        dictvals = {keyval: [np.sum(self.individual_main_roots[keyval][0]
                                   )] for keyval in list(self.individual_main_roots.keys())}

        return dictvals

    
    def single_root_images(self,**kwargs):
        root_dict = {}
        root_cntrs = {}
        root_countours, filtered_image = self.finding_root_countours(**kwargs)
        for i, cnt in enumerate(root_countours):
            imginzeros = np.zeros(filtered_image.shape[:2], dtype=np.uint8)
            x,y,w,h = cv2.boundingRect(cnt)
            contour = np.zeros((filtered_image.shape[:2]), dtype=np.uint8)
           
            cv2.drawContours(contour, [cnt], 0, 255, 1)
            cv2.fillPoly(contour, pts=[cnt], color=(255, 0, 0))

            img_filtered = fill_contours([cnt],filtered_image.shape)
            singlerootimg = img_filtered[y:(y+h),x:(x+w)].copy()

            root_dict['root_{}'.format(i+1)] = singlerootimg
            root_cntrs['root_{}'.format(i+1)] = cnt

        self.individual_root_images = root_dict
        self.individual_root_countours = root_cntrs
        
        return root_dict

    def finding_root_countours(self, **kwargs):
        img_filtered, areathresh = self.preprocessing_image(**kwargs)
        count_after_processing = image_counturs(img_filtered, minarea = areathresh[0], maxarea = areathresh[1])
        
        cont_afterc = copy.copy(count_after_processing)
        img_filtered = fill_contours(cont_afterc,self.plate_image().shape)

        newcont_after_fill = image_counturs(img_filtered[:,:,0], minarea = areathresh[0], maxarea = areathresh[1])
        root_image = copy.copy(img_filtered)
        
        return newcont_after_fill, root_image
        
    def preprocessing_image(self, **kwargs):
        imgprocessed, areathresh = image_preprocessing(self.plate_image(), **kwargs)
        return imgprocessed, areathresh

    def plate_image(self, 
                    bufferheight = 5,
                    bufferwidth = 20):

        (xmin, xmax), (ymin, ymax) = self.frame_default(applythreshold= False, min_line_length = 600, max_line_gap = 50)

        imgframed = self.root_image[ymin+bufferheight:(ymax-bufferwidth),xmin+bufferheight:(xmax-bufferwidth)].copy()

        return imgframed 
    
    def plate_limits(self):
        (xmin, xmax), (ymin, ymax) = self.frame_default(applythreshold= False, min_line_length = 600, max_line_gap = 50)

        return (xmin, xmax), (ymin, ymax)

    def __init__(self, imagepath, splitintotwo = True, startwith = "top") -> None:
        """
        

        Args:
            imagepath (_type_): _description_
            splitintotwo (bool, optional): _description_. Defaults to True.
            startwith (str, optional): bottom or top. Defaults to "top".
        """
        assert os.path.exists(imagepath)
        self.individual_root_images = None
        self.individual_root_countours = None
        self.individual_main_roots = None
        data = cv2.imread(imagepath)
        
        if splitintotwo:
            heighthalf = data.shape[0]//2
            imgtoprocess = data[heighthalf:].copy() if startwith == "top" else data[:heighthalf].copy()

        self._direction = startwith
        
        self.root_image = copy.deepcopy(imgtoprocess)    
        Frame.__init__(self,self._preprocess_plate_image())
        
        RootDrawing.__init__(self)

