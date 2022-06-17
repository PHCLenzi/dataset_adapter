# Objective of this folder

The objective of this code is to develop a parse to transform the Visdrone dataset images into the MARS dataset format that is compatible with the training algorithm for feature vector generating networks. To create this dataset, the file parser_visdrone_to_mars_format.py will be used.

## Format of the matching dataset from the training algo

source 1 = <http://zheng-lab.cecs.anu.edu.au/Project/project_mars.html>
It is known that the MARS dataset is compatible with this network training algo, so the new dataset (using the VisDrone images) will be based on the MARS dataset structure.

The following is the logic behind the generation of the names.

>Naming Rule of the bboxes
>In bbox "0065C1T0002F0016.jpg", "0065" is the ID of the pedestrian. "C1" denotes the first camera (there are totally 6 cameras).
>"T0002" means the 2th tracklet.
> "F016" is the 16th frame within this tracklet.
>For the tracklets, their names are accumulated for each ID; but for frames, they start from "F001" in each tracklet.
>Using the same convention as Market-1501, ID = "00-1" means junk images which do not affect retrieval accuracy; ID = "0000" means distractors, which negatively affect retrieval accuracy.

source 2 = <https://github.com/liangzheng06/MARS-evaluation>

"

1. Training should be done with images in folder "bbox_train".

2. Bounding box feature extraction should follow the order specified in "root/info/test_name.txt" and "root/info/train_name.txt." The newly extracted feature should be loaded in line 19-20 in "root/test_mars.m""

# MARS dataset train structures

The folder structure of this MARS dataset.

./bbox_train
./bbox_train/0001
./bbox_train/0001/0001C1T0001F001.jpg

Estrutura do nome: <(0001)ID do pedestrian><(C1)used camera ><(T0002)captured tracklet><(F016)number of the frame inside the tracklet>.jpg

# VisDrone Dataset training structure

The data set contains two main folders "VisDrone2019-MOT-train"(7.6 GB) and "VisDrone2019-MOT-val"(1.5 GB), both folders have the same structure.

In "/home/hg128004/VisDroneDataSet/VisDrone2019-MOT-train/" we have the folders "annotations" and "sequences".

In "annotations" we have txt files like "uav0000013_00000_v.txt" which is supposed to be a .csv. This "uav0000013_00000_v.txt" has a series of coordinates like:

This link [here](http://aiskyeye.com/evaluate/results-format_2021/), in multi objects tracking, talks about the sequence of values:

Here:

```
<frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```

```sh
(beguin))
1,0,593,43,174,190,0,0,0,0
2,0,592,43,174,189,0,0,0,0
3,0,592,43,174,189,0,0,0,0
...
.
268,0,624,47,181,185,0,0,0,0
269,0,625,48,180,184,0,0,0,0
1,1,489,307,41,62,0,0,0,0
2,1,489,306,41,62,0,0,0,0
...
.
(middle)
102,25,579,240,20,50,1,1,0,1
103,25,579,240,20,50,1,1,0,1
104,25,579,240,20,50,1,1,0,1
105,25,579,240,20,50,1,1,0,1
106,25,579,241,20,50,1,1,0,1

```

## classes of dataset visdrone

The object category indicates the type of annotated object, (i.e., ignored regions (0), pedestrian (1), people (2), bicycle (3), car (4), van (5), truck (6), tricycle (7), awning-tricycle (8), bus (9), motor (10), others (11)).
Notably, if a human maintains standing pose or walking, we classify it as pedestrian; otherwise, it is classified as a person.

# Algorithm of building a dataset using the MARS structure with the visdrone images

## 1. Excluded unnecessary info from VisDroneDS description txt

Exclude the last 4 information (accuracy, occlusion...) of "uav000NUMBER_00NUMBER_v.txt" (folders in /home/hg128004/VisDroneDataSet/VisDrone2019-MOT-train/annotations/...)

## 2.  add the information of the name of the video (file with the images) from which the image will be captured

We can add the information by joining the video name with the object id and put all this in place of <target_id>.

## 3.  Do this with the next txt of the next video and join it with the previous one

All the lists were concatenated into one

## 4. Generate image files by reading this giant array

### 4.1. for each id you should separate the list of images between training and test group

The classes whose name is an odd number, "0001" for example, should go in the bbox_train folder. Now, clsses with an even-numbered name, "0002", should go in the bbox_test folder.

### 4.2. You should crop the image in the right proportion used in training the MARS net

### 4.3. Save only two sets of images in the right proportion in the same folder structure of MARS dataset

### 4.4. Change and generate "root/info/test_name.txt" and "root/info/train_name.txt."

[Source](https://github.com/liangzheng06/MARS-evaluation#mars-evaluation)

# Environment conda

Conda env visDroneDS
[Build env from another env](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file): local do env original
On this [bitbucket](https://bitbucket.opt01.net/projects/CVD/repos/vis_drone/browse?at=refs%2Fheads%2Ffeature%2Fadd-parser-from-visdrone-format-to-mars-format)
The file conda_environment.yml.

```

conda env create -f <local of clone>/vis_drone/conda_environment.yml
conda activate vis_drone

```

# Running the code of Dataset building

'''
$ <path of conda env>/bin/python <path to de the code>/parser_visdrone_to_mars_format.py -o /home/hg128004/VisDroneDataSet -i /home/hg128004/VisDroneDataSet/VisDrone2019-MOT-train/sequences -a /home/hg128004/VisDroneDataSet/VisDrone2019-MOT-train/annotations -t 100

Exp:
$
/home/hg128004/.conda/envs/vis_drone/bin/python /home/hg128004/Projects/parser_algo_Visdrone_img_into_MARS_format/parser_visdrone_to_mars_format.py -o /home/hg128004/Modified_Datasets/Visdrone_Img_MARS_Format_reshaped_evenOdd_person_car -i /home/hg128004/Originals_Datasets/VisDrone/VisDrone2019-MOT-train/sequences -a /home/hg128004/Originals_Datasets/VisDrone/VisDrone2019-MOT-train/annotations -t 100

'''

The imput of this script: 
 * -i <input_img_foder_path>()
 * -o <output_img_foder_path>(path to the imagedataset annotations, here the path where the masses bbox_test and bbox_train will be created and loaded)
 * -a <input_annotation_foder_path> (path to the imagedataset annotations, here the visdrone)
 * -t <thresholder>(minimum value of images per class. Classes smaller than this value will be ignored and will not be part of the dataset.)
