[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FAugmenting_Detection_Dataset&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
# Augmentation for Detection Dataset

This repo utilizes [**imageaug**](https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html) library to augument the object detection dataset.

## Usage

```
python \path_2_script\bb_aug.py -i=C:\Users\Desktop\new\in -o=C:\Users\Desktop\new\op -c train bus cycle car -iter=3
```
For help
```
python D:\path 2 where file is located\bb_aug.py -h

usage: bb_aug.py [-h] [-i IMG_PATH] [-o OP_IMG_PATH]
                 [-c CLASSES [CLASSES ...]] [--xml_path XML_PATH]
                 [--op_xml_path OP_XML_PATH] [-iter ITERATIONS]

optional arguments:
  -h, --help            show this help message and exit
  -i IMG_PATH, --img_path IMG_PATH
                        path to read images.
  -o OP_IMG_PATH, --op_img_path OP_IMG_PATH
                        path to write images.
  -c CLASSES [CLASSES ...], --classes CLASSES [CLASSES ...]
                        a list containing names of all classes in dataset
  --xml_path XML_PATH   path to read xml files if None then same as img_path.
  --op_xml_path OP_XML_PATH
                        path to write xml files if None then same as
                        op_img_path.
  -iter ITERATIONS, --iterations ITERATIONS
                        Number of times to augment each image e.g. if input
                        dir has 2 images and iterations=4 then op dir will
                        have 8 images, default is 1.
```
## Output
```
============================================================
Images Found =  3
Annot. Found =  3
------------------------------------------------------------
Augmneted Files =  9
============================================================
Augumenting Dataset (iteration 1of3): 100%|████████████████████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.00s/it]
Augumenting Dataset (iteration 2of3): 100%|████████████████████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.23s/it]
Augumenting Dataset (iteration 3of3): 100%|████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.40it/s]
```

## Dependency (Libraries)

1. imgaug
2. matplot
3. cv2
4. xmltodict
5. dicttoxml

### Note:
_____
Repo takes about 1 second(s) to read and write one augumented image and file, for fast processing comment the plotting lines in the code.

## Follwing are some examples of input images with their b_boxes drawn on them and their corresponding augumented outputs.


![alt text](https://github.com/Mr-TalhaIlyas/Augumenting_Detection_Dataset/blob/master/images/Slide1.JPG?raw=true)

![alt text](https://github.com/Mr-TalhaIlyas/Augumenting_Detection_Dataset/blob/master/images/Slide2.JPG?raw=true)

![alt text](https://github.com/Mr-TalhaIlyas/Augumenting_Detection_Dataset/blob/master/images/Slide3.JPG?raw=true)
