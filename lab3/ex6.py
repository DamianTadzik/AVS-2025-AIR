## https://github.com/ozantezcan/BSUV-Net-inference

'''
Exercise 3.6 Run the BSUV-Net convolutional neural network.
 • Download all the files for the neural network from the UPeL platform.
 • The pedestrians database will be used, but appropriately scaled to the size of
 the network input and converted to video sequences,
 • Open infer_config_manualBG.py file and modify the paths to:– inclass SemanticSegmentation specify the absolute path to the segmentation
 folder– variable root_path,– in class BSUVNet specify the path to the network model BSUV-Net-2.0.mdl
 in the trained_models folder– model_path,– in class BSUVNet specify the path to an image that contains only the back
ground– the first image in the set pedestrians– variable empty_bg_path
 • Intheinference.pyfilespecify the path to the network input, i.e. the pedestrians
 video sequence– variable inp_path and a path to where the network output is to
 be stored (path with the file name and file extension)– variable out_path
'''

'''
python inference.py <vid_in> <vid_out>
python BSUV-Net-inference\inference.py D:\Dane\Studia_AiR_semestr_8\advanced_vision_systems\lab3\pedestrians_input.mp4 D:\Dane\Studia_AiR_semestr_8\advanced_vision_systems\lab3\BSUV-Net-inference\inference.py
'''
# Resources are not avaliable

