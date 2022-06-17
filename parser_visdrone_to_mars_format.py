''' to run 
$
<conda env>bin/python <clone local>vis_drone/src/data/parser_visdrone_to_mars_format.py -o /home/hg128004/VisDroneDataSet -i /home/hg128004/VisDroneDataSet/VisDrone2019-MOT-train/sequences -a /home/hg128004/VisDroneDataSet/VisDrone2019-MOT-train/annotations
exp
$ 
/home/hg128004/.conda/envs/vis_drone/bin/python /home/hg128004/vis_drone/src/data/parser_visdrone_to_mars_format.py -o /home/hg128004/Modified_Datasets/Visdrone_Img_MARS_Format_reshaped_evenOdd_person_car -i /home/hg128004/Originals_Datasets/VisDrone/VisDrone2019-MOT-train/sequences -a /home/hg128004/Originals_Datasets/VisDrone/VisDrone2019-MOT-train/annotations -t 100

'''

import getopt
import os
import shutil
import sys
import time

import cv2
import pandas as pd
from matplotlib import pyplot as plt


def myfunc(argv):
    
    global arg_input_path 
    global arg_output_path
    global arg_input_annotation_path 
    global arg_thresholder 

    """
    With the variable 'is_reshape = True' it means that during the cropping and the separation of the images by ID the cropping will be 
    done in order to respect the ratio 256/128 = (height /width) = 2. This will be done to be able to use the original cosine metric 
    learning training algorithm for the MARS dataset.
    [256, 128, 3].
    """
    global is_reshape
    is_reshape = False

    global height
    global width 

    height = 256
    width = 128

    arg_input_path = ""
    arg_output_path = ""
    arg_input_annotation_path = ""

    '''
    the thresholder parameter serves to tell you the minimum value of the amount of images an ID must have to be part of the final 
    dataset. For example, if an object with a unique ID has only 30 frames in its video and the thresh holder is 40, the images of 
    this object will be ignored and will not be part of the final dataset 
    '''



    arg_thresholder = "0"
    # print('argv = ',argv)


    arg_help = "{0} -i <input_img_foder_path> -o <output_img_foder_path> -a <input_annotation_foder_path> -t<thresholder>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hi:u:o:a:t:", ["help", "input_img_foder_path=", "output_img_foder_path=", "input_annotation_foder_path=","thresholder="])
    except:
        print(arg_help+"aqui!")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-i", "--input_img_foder_path"):
            arg_input_path = arg
    
        elif opt in ("-o", "--output_img_foder_path"):
            arg_output_path = arg

        elif opt in ("-a", "--input_annotation_foder_path"):
            arg_input_annotation_path = arg

        elif opt in ("-t", "--thresholder"):
            arg_thresholder = arg

    print('input_img_foder_path:', arg_input_path)
    print('output_img_foder_path:', arg_output_path)
    print('input_annotation_foder_path:', arg_input_annotation_path)
    print('arg_thresholder:', arg_thresholder)

    #unsaved images counter because when resizing the bbox (using the variable is_reshape=True) the limits of the new bbox are outside the image
    global unsaved_imgs_counter
    unsaved_imgs_counter = 0

    # start time counter
    global start
    start = time.time()
    # the following path should contain the path to the visdrone dataset annotation files
    # visdrone_annotations_path = "/home/hg128004/VisDroneDataSet/VisDrone2019-MOT-train/annotations"
    visdrone_annotations_path = arg_input_annotation_path

    list_all_ids_bboxs = integrateTxtFiles(visdrone_annotations_path)

    print_time_counter()

    # Export the list as a csv file
    # list_all_ids_bboxs.to_csv(arg_output_path+'/list_all_ids_bboxs.txt', sep=',', index=True)
    path_to_list_all_ids_bboxs = arg_output_path+'/list_all_ids_bboxs.txt'
    list_all_ids_bboxs.to_csv(path_to_list_all_ids_bboxs, sep=',', index=True)
    
    
    '''
    # debug section
    list_all_ids_bboxs = pd.read_csv("/home/hg128004/Modified_Datasets/Visdrone_Img_into_MARS_Format_reshaped/list_all_ids_bboxs.txt", sep=",", header=0)
    print("list_all_ids_bboxs =" ,list_all_ids_bboxs)
    #end debug section
    '''
    


    # set the folder path of the training and test images
    # the following path should contain the path where the image files separated by object id will be saved
    PATH_bbox_target = arg_output_path
    PATH_bbox_train = PATH_bbox_target+"/bbox_train"
    PATH_bbox_test = PATH_bbox_target+"/bbox_test"
    
    # defines the path of the original images 
    # the following path must contain the path of the folders with the original image sequences from the visdrone dataset
    PATH_VisDro_imgs_orig = arg_input_path

   
    # reset image destination folder
    reset_folders_image(PATH_bbox_train,PATH_bbox_test)
    print_time_counter()

    image_separator_by_id(list_all_ids_bboxs,PATH_bbox_train,PATH_bbox_test,PATH_VisDro_imgs_orig)
    print_time_counter()
    print("End of the process ")




def print_time_counter():
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("algorithm running time : {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def integrateTxtFiles(path_to_folder_annotations):
    ## runs the whole path from path_to_folder_annotations and generates a list with the names of all the annotation files
    list_of_annotation_files_names = []
    for file in os.listdir(path_to_folder_annotations):
        if file.endswith(".txt"):
            # generates a list with the names of the annotations without the ".txt" extention
            list_of_annotation_files_names.append(file[:-4])
    print('len(list_of_annotation_files_names) = ',len(list_of_annotation_files_names))
    print('...')
    ## runs the entire list of annotated txt file names
    # create an empty pandas array
    concatenated_array = pd.DataFrame()
    debug_index = 0
    while debug_index < len(list_of_annotation_files_names):
        # read a single txt file and turn it into a pandas array
        data_raw = pd.read_csv(path_to_folder_annotations+'/'+list_of_annotation_files_names[debug_index]+".txt", sep=",", header=None)
        # define the name of the Pandas Array columns
        data_raw.columns = ["frame_index", "target_id", "bbox_left", "bbox_top", "bbox_width", "bbox_height", "score", "object_category", "truncation", "occlusion"]
        # filter and leave only the information of the bbox that belong to the classes of people, pedestrians and cars.
            # reduce. just use the classes people, pedestrians or car in "object class" = 1,2,4
        not_interres_index_class = [0,3,5,6,7,8,9,10,11]
        for i in not_interres_index_class:
            data_raw = data_raw[ data_raw["object_category"]!=i]
        
        # remove the columns ['score', 'object_category', 'truncation', 'occlusion']
        data_raw = data_raw[["frame_index", "target_id", "bbox_left", "bbox_top", "bbox_width", "bbox_height"]]

        ## get the name of corresponding images collection and integrates with target_id info column
        # for loop to run all the row of the array
        for index in data_raw.index:
            target_id = data_raw["target_id"][index]
            data_raw["target_id"][index] = list_of_annotation_files_names[debug_index]+"_"+str(target_id)

        # concatenate arrays
        concatenated_array=pd.concat([concatenated_array,data_raw])
        debug_index+=1
        progress_value = float("{:.2f}".format((debug_index/len(list_of_annotation_files_names))*100))




    # filter threshold, IDs with an img amount less than the threshold won't integrate the dataset
    list_of_ids_names = concatenated_array['target_id'].unique()

    print("debug:: start the filter threshold!")
    # verify the amount of images of each ID
    debug_index_aux = 0

    for id in list_of_ids_names:
        debug_index_aux += 1
        unique_id_array = concatenated_array[concatenated_array['target_id']==id]
        if(len(unique_id_array.index)< int(arg_thresholder)):
            concatenated_array = concatenated_array[concatenated_array['target_id']!=id]
    print("debug:: finished the filter threshold!")

    # Generates a list of identified ids 
    print('At the end,',debug_index,' lists were concatenades')
    return concatenated_array

# Restart blank directories
def reset_folders_image(PATH_bbox_train,PATH_bbox_test):
    
    if os.path.isdir(PATH_bbox_train):
        shutil.rmtree(PATH_bbox_train)
    if os.path.isdir(PATH_bbox_test):
        shutil.rmtree(PATH_bbox_test)

    os.mkdir(PATH_bbox_train)
    os.mkdir(PATH_bbox_test)

# function that receives the image path, cuts out the selected part and saves it in the specified path
def crop_an_img_and_save_it(origin_img_path, bbox_info, goal_img_path_with_name,is_reshape):
    global unsaved_imgs_counter
    
    img = cv2.imread(origin_img_path)
    '''
        bbox_info[0]=(bbox_left)(x)
        bbox_info[1]=(bbox_top)(y)
        bbox_info[2]=(bbox_width)(dx)
        bbox_info[3]=(bbox_height)(dy)
        crop_img = img[y:y+h, x:x+w]
    '''
    crop_img = img[bbox_info[1]:bbox_info[3], bbox_info[0]:bbox_info[2]]
    
    if(is_reshape):
        #height = bbox_info[3]-bbox_info[1]
        #width = bbox_info[2]-bbox_info[0]
        crop_img = cv2.resize(crop_img, (width,height), interpolation= cv2.INTER_NEAREST)

    try:
        RGB_im = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        # saves the image to the path: goal_img_path_with_name
        cv2.imwrite(goal_img_path_with_name,crop_img)
        # plt.imshow(RGB_im)
        # plt.show()

    except:
        unsaved_imgs_counter += 1
        print("unsaved_imgs_counter= {}".format(unsaved_imgs_counter))
        print("image:{},does not save".format(goal_img_path_with_name))




def image_separator_by_id(list_all_ids_bboxs,PATH_bbox_train,PATH_bbox_test,PATH_VisDro_imgs_orig):
    # Generates a list of identified ids 
    list_of_ids_names = list_all_ids_bboxs['target_id']
    print("debug::list_of_ids_names = ",list_of_ids_names)


    list_of_ids_names = list_all_ids_bboxs['target_id'].unique()
    # for each id name create a mini array with the bbox info
    debug_aux_index = 0

    for name in list_of_ids_names:
        # prints the progress of the code for monitoring purpose
        progress_value = float("{:.2f}".format((debug_aux_index/len(list_of_ids_names))*100))
        print('building dataset...',progress_value,'%' )
        if(debug_aux_index//400==0):
            print_time_counter()


        one_id_name_array = list_all_ids_bboxs[list_all_ids_bboxs['target_id']== name]
        # If the class folder name is an odd number, it goes to bbox_train. If it's even, it goes to the bbox_test folder
        if(debug_aux_index%2==0):
            id_folder_imgs_name = PATH_bbox_test +'/'+ str(debug_aux_index).zfill(4)
        else:
            id_folder_imgs_name = PATH_bbox_train +'/'+ str(debug_aux_index).zfill(4)
            
        os.mkdir(id_folder_imgs_name)
        # run all row of a array of just one unique id
        for index, row in one_id_name_array.iterrows():
        
            path_of_img = PATH_VisDro_imgs_orig + '/' + str(row['target_id'])[0:18] + '/'+ str(row['frame_index']).zfill(7) + '.jpg'
            bbox_info = (row['bbox_left'],row['bbox_top'],row['bbox_left']+row['bbox_width'],row['bbox_top']+row['bbox_height'])
            ## create a name for img
            # <(0001)ID do pedestrian><(C1)used camera ><(T0002)captured tracklet><(F016)number of the frame inside the tracklet>.jpg
            img_name = str(row['target_id'])[19:].zfill(4) + 'C1' + 'T0001' + 'F' + str(row['frame_index']).zfill(3)+ '.jpg'
            crop_an_img_and_save_it(path_of_img,bbox_info,id_folder_imgs_name+'/'+ img_name,True)

            # break
        debug_aux_index+=1
        
        # print('debug_aux_index = ', debug_aux_index)
        # stop just for debug reasons
        # if (debug_aux_index>35):
        #     break

    # 7702 class in 430 minutes(7h and 10 min)
    print(' dataset completed ')

if __name__ == "__main__":
    myfunc(sys.argv)
