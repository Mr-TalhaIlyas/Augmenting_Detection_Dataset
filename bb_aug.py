import argparse, os
'''
Usage:
    

'''
parser = argparse.ArgumentParser()
# positional required args
parser.add_argument("-i", "--img_path",  help="path to read images.", type=str)
parser.add_argument("-o", "--op_img_path",  help="path to write images.", type=str)
parser.add_argument("-c", "--classes", nargs="+", help="a list containing names of all classes in dataset")
#optional args
parser.add_argument("--xml_path",  help="path to read xml files if None then same as img_path.")
parser.add_argument("--op_xml_path",  help="path to write xml files if None then same as op_img_path.")
parser.add_argument("-iter", "--iterations",  help="Number of times to augment each image \
                    e.g. if input dir has 2 images and iterations=4 then op dir \
                    will have 8 images, default is 1.", type=int, default=1)

args = parser.parse_args()

img_path = args.img_path
op_img_path = args.op_img_path

img_path = img_path.replace('\\', '/') + '/'
op_img_path = op_img_path.replace('\\', '/') + '/'

if args.xml_path:
    xml_path = args.xml_path
    xml_path = xml_path.replace('\\', '/') + '/'
else: 
    xml_path = img_path

if args.op_xml_path:
    op_xml_path = args.op_xml_path
    op_xml_path = op_xml_path.replace('\\', '/') + '/'
else: 
    op_xml_path = op_img_path

iterations = args.iterations

classes = args.classes
#%%
import imgaug as ia
# imgaug uses matplotlib backend for displaying images
#%matplotlib inline
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
# imageio library will be used for image input/output
import imageio
import numpy as np
from tqdm import trange
import cv2
import os, re
import glob
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
import matplotlib.pyplot as plt
import xmltodict
# this library is needed to read XML files for converting it into CSV
import dicttoxml
from xml.dom.minidom import parseString

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def remove_duplicate(s):
    '''
    I created this function b/c i am looping over all the keys in the xml file
    and in case if xmldict have only one object the loop sitll consider the 
    <part> keys to be part of object so i am just looping over same object again 
    and again. So i just rmove those duplication with this
    '''
    x = s.split(' ')
    y = list(set(x))
    y = ' '.join(map(str, y))
    return y
def add_to_contrast(images, random_state, parents, hooks):
    '''
    A custom augmentation function for iaa.aug library
    The randorm_state, parents and hooks parameters come
    form the lamda iaa lib**
    '''
    images[0] = images[0].astype(np.float)
    img = images
    value = random_state.uniform(0.75, 1.25)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    ret = img[0] * value + mean * (1 - value)
    ret = np.clip(img, 0, 255)
    ret = ret.astype(np.uint8)
    return ret
# randomly choose positions in image
positions = ['center','left-top', 'left-center', 'left-bottom', 'center-top',
             'center-bottom', 'right-top', 'right-center', 'right-bottom']

img_path = glob.glob(os.path.join(img_path, '*.jpg')) + \
            glob.glob(os.path.join(img_path, '*.png')) 
img_path = sorted(img_path, key=numericalSort)

xml_path = glob.glob(os.path.join(xml_path, '*.xml'))
xml_path = sorted(xml_path, key=numericalSort)
print('='*60)
print('Images Found = ', len(img_path))
print('Annot. Found = ', len(xml_path))
print('-'*60)
print('Augmneted Files = ', (len(xml_path)*iterations))
print('='*60)
#%%
sometimes = lambda aug: iaa.Sometimes(0.97, aug)
pos = np.random.randint(0, 9)
''' Geometrical Data Aug'''
seq_1 = iaa.Sequential(
        [
        # apply only 2 of the following
        iaa.SomeOf(2, [
            sometimes(iaa.Fliplr(0.9)),
            sometimes(iaa.Flipud(0.9)),
            sometimes(iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=1, backend="cv2")),
            sometimes(iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, order=1, backend="cv2")),
            sometimes(iaa.Affine(rotate=(-25, 25), order=1, backend="cv2")),
            sometimes(iaa.Affine(shear=(-8, 8), order=1, backend="cv2")),
            iaa.OneOf([
            sometimes(iaa.KeepSizeByResize(
                                 iaa.Crop(percent=(0.05, 0.25), keep_size=False),
                                 interpolation='linear')),
            sometimes(iaa.KeepSizeByResize(
                                iaa.CropToFixedSize(width=512, height=512, position=positions[pos]),
                                interpolation="linear"))
            ]),
            ], random_order=True),
        ], random_order=True)
''' Noisy Data Aug'''
seq_2 = iaa.Sequential(
        [
        iaa.OneOf(
            [   
            # Blur each image using a median over neihbourhoods that have a random size between 3x3 and 7x7
            sometimes(iaa.MedianBlur(k=(3, 7))),
            # blur images using gaussian kernels with random value (sigma) from the interval [a, b]
            sometimes(iaa.GaussianBlur(sigma=(0.0, 1.0))),
            sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5))
            ]
        ),
        iaa.Sequential(
            [
            sometimes(iaa.AddToHue((-8, 8))),
            sometimes(iaa.AddToSaturation((-20, 20))),
            sometimes(iaa.AddToBrightness((-26, 26))),
            sometimes(iaa.Lambda(func_images = add_to_contrast))
            ], random_order=True)
        ], random_order=True)
#%%
for p in range(iterations): # how many times to apply random augmentations
    for idx in trange(len(img_path), desc='Augumenting Dataset (iteration {}of{})'.format(p+1, iterations)):
        
        img = cv2.imread(img_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filepath = xml_path[idx]
        
        full_dict = xmltodict.parse(open( filepath , 'rb' ))
        
        # Extracting the coords and class names from xml file
        names = []
        coords = []
        
        obj_boxnnames = full_dict[ 'annotation' ][ 'object' ] # names and boxes
        file_name = full_dict[ 'annotation' ][ 'filename' ]#full_dict[ 'annotation' ][ 'filename' ]
        
        for i in range(len(obj_boxnnames)):
            # 1st get the name and indices of the class
            try:
                obj_name = obj_boxnnames[i]['name']
            except KeyError:
                obj_name = obj_boxnnames['name']  # if the xml file has only one object key
                
            obj_ind = [i for i in range(len(classes)) if obj_name == classes[i]] # get the index of the object
            obj_ind = int(np.array(obj_ind))
            # 2nd get tht bbox coord and append the class name at the end
            try:
                obj_box = obj_boxnnames[i]['bndbox']
            except KeyError:
                obj_box = obj_boxnnames['bndbox'] # if the xml file has only one object key
            bounding_box = [0.0] * 4                    # creat empty list
            bounding_box[0] = int(float(obj_box['xmin']))# two times conversion is for handeling exceptions 
            bounding_box[1] = int(float(obj_box['ymin']))# so that if coordinates are given in float it'll
            bounding_box[2] = int(float(obj_box['xmax']))# still convert them to int
            bounding_box[3] = int(float(obj_box['ymax']))
            bounding_box.append(obj_ind) 
            bounding_box = str(bounding_box)[1:-1]      # remove square brackets
            bounding_box = "".join(bounding_box.split())
            names.append(obj_name)
            coords.append(bounding_box)
        #%
        coords = ' '.join(map(str, coords))# convert list to string
        coords = remove_duplicate(coords)
        coords = coords.split(' ')
        t = []
        for i in range(len(coords)):
            t.append(coords[i].split(','))
        t = np.array(t).astype(np.uint32)
        
        coords = t[:,0:4]
        class_idx = t[:,-1]
        class_det = np.take(classes, class_idx)
        
        #%
        
        #ia.seed(1)
        # for giving one box
        # bbs = BoundingBoxesOnImage([
        #     BoundingBox(x1=coords[0], x2=coords[2], y1=coords[1], y2=coords[3])
        # ], shape=img.shape)
        
        bbs = BoundingBoxesOnImage.from_xyxy_array(coords, shape=img.shape)
        for i in range(len(bbs)):
            bbs[i].label = class_det[i]
            
        #ia.imshow(bbs.draw_on_image(img, color=[255, 0, 0], size=10))
        '''
        Start applying augmentations
        '''
        num = np.random.randint(1, 100)
        if (num % 2) == 0:
            image_aug, bbs_aug = seq_1.augment(image=img, bounding_boxes=bbs)
        elif (num % 2) != 0:
            image_aug, bbs_aug = seq_2.augment(image=img, bounding_boxes=bbs)
        #   disregard bounding boxes which have fallen out of image pane   
        bbs_aug = bbs_aug.remove_out_of_image()
        #   clip bounding boxes which are partially outside of image pane
        bbs_aug = bbs_aug.clip_out_of_image()
        
        #ia.imshow(bbs_aug.draw_on_image(image_aug, size=5))
    
        '''
        Now updata the dictionary wiht new augmented values
        '''
        
        #full_dict = xmltodict.parse(open( filepath , 'rb' ))
        
        obj_boxnnames = full_dict[ 'annotation' ][ 'object' ] # names and boxes
        full_dict[ 'annotation' ][ 'filename' ] = str("{}_aug_{}".format(p,file_name))
        full_dict[ 'annotation' ][ 'path' ] = str('None')
        
        for i in range(len(bbs_aug)):
            # 1st get the name and indices of the class
            try:
                obj_boxnnames[i]['name'] = str(bbs_aug[i].label)
            except KeyError:
                obj_boxnnames['name']   = str(bbs_aug[i].label)# if the xml file has only one object key
            obj_ind = [i for i in range(len(classes)) if obj_name == classes[i]] # get the index of the object
            obj_ind = int(np.array(obj_ind))
            # 2nd get tht bbox coord and append the class name at the end
            try:
                obj_boxnnames[i]['bndbox']['xmin'] = str(int(bbs_aug[i][0][0]))
                obj_boxnnames[i]['bndbox']['ymin'] = str(int(bbs_aug[i][0][1]))
                obj_boxnnames[i]['bndbox']['xmax'] = str(int(bbs_aug[i][1][0]))
                obj_boxnnames[i]['bndbox']['ymax'] = str(int(bbs_aug[i][1][1]))
            except KeyError:
                obj_boxnnames['bndbox']['xmin'] = str(int(bbs_aug[i][0][0]))
                obj_boxnnames['bndbox']['ymin'] = str(int(bbs_aug[i][0][1]))
                obj_boxnnames['bndbox']['xmax'] = str(int(bbs_aug[i][1][0]))
                obj_boxnnames['bndbox']['ymax'] = str(int(bbs_aug[i][1][1]))
        
        '''
        Delete the excess objects which were in the original dict, because we are 
        using the orginal dict to rewrite the annotations
        '''
        try:
            del(full_dict['annotation']['object'][len(bbs_aug):])
        except TypeError:
            pass
        '''
        Now write the new augmented xml file and image
        '''
        # dictionary to xml
        xml = dicttoxml.dicttoxml(full_dict, attr_type=False) # set attr_type to False to not wite type of each entry
        # xml bytes to string
        xml = xml.decode() 
        # parsing string
        dom = parseString(xml)
        # pritify the string
        dom = dom.toprettyxml()
        # remove the additional root added by the library
        dom = dom.replace('<root>','')
        dom = dom.replace('</root>','')
        if dom.find('<item>') != -1: 
            dom = dom.replace('<object>','')
            dom = dom.replace('</object>','')
            dom = dom.replace('<item>','<object>')
        dom = dom.replace('</item>','</object>')
        # write the pretified string
        xmlfile = open(op_xml_path + "{}_aug_{}.xml".format(p,file_name[:-4]), "w") 
        xmlfile.write(dom) 
        xmlfile.close() 
        # wirte image
        imageio.imwrite(op_img_path + '{}_aug_{}'.format(p,file_name), image_aug)    
