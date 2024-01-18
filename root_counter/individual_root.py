import numpy as np
from plantcv import plantcv as pcv 
from root_counter.plt_functions.utils import random_color_generator
import cv2
from skimage.filters import meijering
from skimage.morphology import skeletonize, thin



def change_whereformat(positionwhere):
    listpos = []
    for i in range(len(positionwhere[0])):
        listpos.append([positionwhere[0][i],positionwhere[1][i]])
    return np.array(listpos)

def rltb_boundaries(image, centercoords, kernelsize = (1,1)):
    yp, xp = centercoords
    heightk, widthk = kernelsize[0], kernelsize[1]
    ## evaluate coords
    bottom = yp-heightk if yp-heightk >= 0 else 0
    left = xp-heightk if xp-widthk >= 0 else 0

    top = yp+heightk+1 if yp+heightk+1 <= image.shape[0]  else image.shape[0]
    right = xp+widthk+1 if xp+widthk+1 <= image.shape[1] else image.shape[1]

    return right, left, top, bottom

def chunk_matrix(image, centercoords, kernelsize = (1,1)):
    
    right, left, top, bottom = rltb_boundaries(image, centercoords, kernelsize = kernelsize)
    matrixchunck = image[bottom:top,left:right]

    return matrixchunck

def chunk_coords(image, centercoords, kernelsize = (1,1)):
    x = np.arange(0,image.shape[1], 1)
    y = np.arange(0,image.shape[0], 1)

    right, left, top, bottom = rltb_boundaries(image, centercoords, kernelsize = kernelsize)

    distchunkx = x[left:right]
    distchunky = y[bottom:top]

    return distchunky, distchunkx

def check_positions(image, starts_with = 'top'):
    onepos = np.where(image == 1)

    if starts_with == 'top':
        ypi = onepos[0][0]
        xpi = onepos[1][0]
    if starts_with == 'left':
        lpos = np.argmin(onepos[1])

        ypi=onepos[0][lpos]
        xpi=onepos[1][lpos]
    if starts_with == 'right':
        lpos = np.argmax(onepos[1])
        ypi, xpi =onepos[0][lpos],onepos[1][lpos]
    if starts_with == 'bottom':
        lpos = np.argmax(onepos[0])
        ypi, xpi =onepos[0][lpos],onepos[1][lpos]
    
    return ypi, xpi

def find_root_tip(image, all_options = ['top', 'left','right','bottom']):
    assert len(all_options)>0
    
    wherepos = []

    posopt = 0
    alreadyprocessed = []
    while len(wherepos) != 2:
        #remoptions = all_options[1:]
        alreadyprocessed.append(all_options[posopt])
        ypi, xpi = check_positions(image, starts_with = all_options[posopt])

        snipimage = chunk_matrix(image, (ypi, xpi), kernelsize = (1,1))
        wherepos = change_whereformat(np.where(snipimage==1))
        #print(wherepos)
        posopt = posopt+1

    return (ypi, xpi), alreadyprocessed


def reconstruct_main_root(rootimg, pathcoords):
    reconstruct = np.zeros(rootimg.shape)
    for i in pathcoords:
        yp,xp = i.split('-')
        reconstruct[int(yp), int(xp)] = 1
        
    return reconstruct

def find_following_point(img, initcoords, rootcoords, kernelsize = (1,1)):

    ypos,xpos = initcoords
    
    matrixchunck = chunk_matrix(img, (ypos,xpos), kernelsize)
    distchunky, distchunkx = chunk_coords(img, (ypos,xpos), kernelsize)

    wherepos = change_whereformat(np.where(matrixchunck==1))
    newpos = [np.array([distchunky[i[0]],distchunkx[i[1]]]) for i in list(wherepos) if '{}-{}'.format(distchunky[i[0]],distchunkx[i[1]]) not in rootcoords]

    return newpos

def upto_bif(img, initcoords, maxdistance = None,kernelsize = (1,1)):
    #def finding_main_root(npone)
    x = np.arange(0,img.shape[1], 1)
    y = np.arange(0,img.shape[0], 1)

    ypos,xpos = initcoords
    rootpathcoords = ['{}-{}'.format(y[ypos],x[xpos])]
    
    newpos = find_following_point(img, (ypos,xpos), rootpathcoords, kernelsize = kernelsize)

    interrupt = False
    while not interrupt:
        if len(newpos) == 1:
            ypos = newpos[len(newpos)-1][0]
            xpos = newpos[len(newpos)-1][1]

            rootpathcoords.append('{}-{}'.format(y[ypos],x[xpos]))

            newpos = find_following_point(img, (ypos,xpos), rootpathcoords, kernelsize = kernelsize)
        else:
            interrupt = True
            lastpositionpath = newpos
        if maxdistance:
            if len(rootpathcoords)>maxdistance:
                interrupt = True
                lastpositionpath = newpos
                
    return rootpathcoords, lastpositionpath



def initialize_draw_root(img, kernelsize = (1,1), tip_options=['top', 'left', 'right', 'bottom']):

    (ypos,xpos),tipoptionsprocessed = find_root_tip(img, all_options=tip_options)

    main_rootcoords, nextpath = upto_bif(img, (ypos,xpos), maxdistance = None,kernelsize = kernelsize)

    return main_rootcoords, nextpath, tipoptionsprocessed

## bifurcation check which bigger

def check_bifurcation(img, main_rootcoords, bif_coords, maxdistance = 200, kernelsize = (1,1)):
    matrixtomod = img.copy()
    mainroot = reconstruct_main_root(matrixtomod,main_rootcoords)

    typeofroot = {'main':[],
                'lateral': []}
    rmvpos = 0
    bifextensions = []
    lastpoints = []
    for rmvpos in range(len(bif_coords)):

        rootleft = matrixtomod- mainroot
        rootleft[bif_coords[rmvpos][0],bif_coords[rmvpos][1]] = 0

        bif_initialpos = [bif_coords[i] for i in range(len(bif_coords)) if i != rmvpos]
        possiblepath, lastpoint = upto_bif(rootleft, bif_initialpos[0], maxdistance = maxdistance, kernelsize=kernelsize)
        bifextensions.append(possiblepath)
        lastpoints.append(lastpoint)

    # check extension two criteria must accompplish the extension mus be the highest and it must finish
    
    if len(bifextensions) == 2:
        if len(bifextensions[0]) >= len(bifextensions[1]):
            mainrootpos = 0 if len(lastpoints[0]) != 0 else 1
        else:
            mainrootpos = 1 if len(lastpoints[1]) != 0 else 0
    else:

        mainrootpos = np.argmin([len(bifextensions[i]) for i in range(len(bifextensions))])

    ## todo what would happen if the main is shorter
    typeofroot['main'] = bifextensions[mainrootpos] 
    typeofroot['lateral'] = bifextensions[mainrootpos-1] 

    return typeofroot, lastpoints[mainrootpos]

    

class RootDrawing(object):
    @staticmethod
    def preprocess_single_root(imgcopy, prune = True, prunesize = 150):
        
        #imgcopy = self.singlerootimg.copy()

        ## skeletonize
        skimg = (skeletonize(imgcopy/255.)*255).astype(np.uint8)
        npones = np.ones(skimg.shape)
        npones[skimg == 0] = 0

        if prune:
            skimg= pcv.morphology.prune(skel_img=skimg[:,:,0], size=prunesize)
    
        npones = np.ones(skimg[0].shape)
        npones[skimg[0] == 0] = 0
        return npones


    def detect_main_root(self, image=None, max_lateralrootdist = 300):
        root_dict = {'main':[],
                     'lateral': []}
        
        self.singlerootimg = image

        imgcopy = self.preprocess_single_root(self.singlerootimg.copy())
        
        startingdirections = ['top', 'left', 'right', 'bottom']


        inittip = {}
        initopt = 0
        try:

            while len(startingdirections) > 1:

                max_lateralrootdist = 300

                imgcopy = self.preprocess_single_root(self.singlerootimg.copy(),prunesize=150)

                listalreadypos, last_coord, tipoptionsprocessed = initialize_draw_root(imgcopy, kernelsize=(1,1),tip_options= startingdirections[initopt:])

                root_dict['main'] = listalreadypos

                while len(last_coord)>0:
                    mxdist = max_lateralrootdist if len(last_coord)>1 else None

                    listalreadypos = root_dict['main']
                    if mxdist is not None:
                        root_coordsaftbif, last_coord =  check_bifurcation(imgcopy, listalreadypos,
                                                                            last_coord, maxdistance = mxdist, kernelsize = (1,1))
                        root_dict['main'] = root_dict['main'] + root_coordsaftbif['main']
                        root_dict['lateral'].append(root_coordsaftbif['lateral'])

                    if mxdist is None:
                        mainroot_tmp = reconstruct_main_root(imgcopy,listalreadypos)
                        rootleft = imgcopy- mainroot_tmp.copy()
                        root_coordsaftbif, last_coord =  upto_bif(rootleft, last_coord[0], maxdistance = mxdist, kernelsize=(1,1))
                        root_dict['main'] = root_dict['main'] + root_coordsaftbif

                mainroot_tmp = reconstruct_main_root(imgcopy,root_dict['main'])
                inittip[tipoptionsprocessed[len(tipoptionsprocessed)-1]] = [mainroot_tmp,root_dict]
                startingdirections= [opt for opt in startingdirections if opt not in tipoptionsprocessed]
                #print('after',startingdirections)
                #print(np.sum(mainroot_tmp))
        except:
            pass

        maxlength = np.argmax([np.sum(inittip[dir][0]) for dir in list(inittip.keys())])

        mainroot_tmp = inittip[list(inittip.keys())[maxlength]][0]

        return mainroot_tmp, inittip[list(inittip.keys())[maxlength]][1]
        

    def __init__(self) -> None:
        
        pass
