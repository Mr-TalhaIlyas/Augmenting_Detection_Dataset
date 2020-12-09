

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
import os
import glob
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
import matplotlib.pyplot as plt
import xmltodict
# this library is needed to read XML files for converting it into CSV
import dicttoxml
from xml.dom.minidom import parseString

#classes = [ "blossom_end_rot", "graymold","powdery_mildew","spider_mite","spotting_disease"]
classes = [ "mitosis", "non_mitosis","apoptosis","tumor","non_tumor", "lumen", "non_lumen"]
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

#%
img_path = glob.glob(os.path.join('../Datasets/Breast_Biopsy/xml_boxes/img/*.jpg')) 
xml_path = glob.glob(os.path.join('../Datasets/Breast_Biopsy/xml_boxes/xml_new/*.xml'))

print('Images Found = ', len(img_path))
print('Annot. Found = ', len(xml_path))

op_img_path = '../Breast_Biopsy/aug_data/img/'
op_xml_path = '../Breast_Biopsy/aug_data/xml/'

#%%
print('='*60)
print('WARNING: The "imgaug" library is quite slow in augumenting \nthe detection dataset. \nIt takes about 2~3 Seconds per image\
      \nTurn off plotting to speed-up.')
print('='*60)

#for p in range(9):
for idx in trange(len(img_path), desc='Augumenting Dataset'):

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
    ''' Max Data Aug'''
    seq_1 = iaa.Sequential([
        # these are essential augumentations
        iaa.Fliplr(0.5),
        iaa.Flipud(0.3),
        iaa.Crop(px=(0, 16)),

        # these are less important means only a few of the following will be applied
        iaa.SomeOf((0, 6), [

        iaa.GammaContrast(1.5),
        iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}, scale=(0.5, 0.9)),
        iaa.Affine(rotate=(-60, 60)),
        iaa.Multiply((0.5, 1.5)),
        iaa.AdditiveGaussianNoise(scale=(10, 60)),
        iaa.GaussianBlur(sigma=(0, 0.5))
        ],random_order=True),

        ], random_order=True)
    ''' Optmized Data Aug'''
    seq_2 = iaa.Sequential([

        iaa.Fliplr(0.6),
        iaa.Flipud(0.4),
        iaa.Affine(rotate=(-25, 25)),
        iaa.Crop((200, 400), keep_size=True)
        # some of the follwing will be applied
        # iaa.SomeOf((0, 3), [


        # iaa.Crop(px=(0, 70)),
        # iaa.Affine(rotate=(-25, 25)),
        # iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})

        # ],random_order=True),

        ], random_order=True)
    ''' SSD Upscale Crop'''
    # randomly choose positions in image
    positions = ['center','left-top', 'left-center', 'left-bottom', 'center-top',
                 'center-bottom', 'right-top', 'right-center', 'right-bottom']
    SSD = iaa.Sequential([

        iaa.KeepSizeByResize(iaa.CropToFixedSize(width=470, height=370, position=positions[0]), interpolation="linear")

        ], random_order=True)
    #%

    #for i in range(10):
    image_aug, bbs_aug = SSD.augment(image=img, bounding_boxes=bbs)
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
    full_dict[ 'annotation' ][ 'filename' ] = str("aug_{}.jpg".format(file_name[:-4]))
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
    dom = dom.replace('<object>','')
    dom = dom.replace('</object>','')
    dom = dom.replace('<item>','<object>')
    dom = dom.replace('</item>','</object>')
    # write the pretified string
    xmlfile = open(op_xml_path + "aug_{}.xml".format(file_name[:-4]), "w") 
    xmlfile.write(dom) 
    xmlfile.close() 
    # wirte image
    imageio.imwrite(op_img_path + 'aug_{}'.format(file_name), image_aug)

        
        
        
        
        
