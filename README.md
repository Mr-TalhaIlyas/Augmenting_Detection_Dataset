# Augumentation for Detection Dataset

This repo utilizes [**imageaug**](https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html) library to augument the object detection dataset.

## Input

Directory containing image and xml files

## Output

Directory containing augumented image and xml files

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
