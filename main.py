from root_counter.root_delimitation import RootImage
import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt

def check_existance(func):
    def inner(path):
        if not os.path.exists(path):
            raise ValueError(f"Unable to open the specified path: {path}")
        return func(path)
    
    return inner
        

@check_existance
def read_configuration(path):
    
    if path.endswith('txt'):
        with open('readme.txt') as f:
            paths = f.readlines(f)
            
    if path.endswith('yaml'):
        with open(path) as f:
            paths = yaml.safe_load(f)
    
    return paths

def chek_folder(path):
    os.mkdir(path) if not os.path.exists(path) else None

def export_image(img, path, figsize = (15, 10)):
    
    plt.figure(figsize=figsize)     
    plt.imshow(img)
    plt.savefig(path)
    print(f"image saved in {path}")
    plt.close()


def define_output_path(path, outputpath):
    
    ## get filename
    fn = os.path.basename(path)
    
    chek_folder(outputpath)
    
    return os.path.join(outputpath, fn)
    

def count_roots_single_image(image_path, outputpath, exportimage = True):

    imagesdf = []
    for direction in ['top','bottom']:
        rootdetector  = RootImage(os.path.join(image_path),startwith=direction)
        roots = rootdetector.single_root_images(min_thresh = 80)
        imgroot = rootdetector.plot_roots_over_image()
        
        rootvals = rootdetector.find_all_lengths()

        ## export image
        imgfn = define_output_path(image_path, outputpath)
        imgfn = imgfn.replace('.jpg','_'+direction+'.jpg')
        if exportimage:
            export_image(imgroot, 
                         imgfn, figsize = (15, 10))
        
        ## save as df
        df = pd.DataFrame(rootvals).T
        df['fn'] = imgfn
        df['position'] = direction
        
        imagesdf.append(df)
        
    return imagesdf

    
    

def main():
    
    paths = read_configuration('configuration.yaml')

    fn_images = [i for i in os.listdir(paths['paths']['inputs']) if i.endswith('.jpg')]
    dfdata = []
    for fn in fn_images:
        image_path = os.path.join(paths['paths']['inputs'], fn)
        print(image_path)
        imgdf = count_roots_single_image(image_path, paths['paths']['outputs'], 
                                 exportimage = True)
         
        dfdata.append(imgdf[0])
        dfdata.append(imgdf[1])
    
    ## export file
    pd.concat(dfdata).to_csv(os.path.join(
        paths['paths']['outputs'],'root_length.csv'))
    

if __name__ == "__main__":
    main()


