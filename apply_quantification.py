import os
import numpy as np
import pandas as pd
import tifffile
from utils import listdirNHF, form_mask_from_roi, maintain_exclude_image_rois, match_arrays_dimensions
from multi_channels_multi_quantifications import quantify_channels

class sample_quantifier():

    def __init__(self, analysis_axis=None, roi_structure=None,
                shuffle_times=0, no_quantification_valu_e=np.nan, channels_binarization_thresholds=0, transform_to_label_img=False,
                get_mask_area_val_4zero_regionprops=0, count_regions_number_threshold_roi_mask=0, n_of_region_4areas_measure=0,
                reg_eucl_dist_within_arr_val_n_regions_nopass=1, get_convex_hull_min_px_num=2, min_px_over_thresh_common=-1, measure_pixels_overlap_n_px_thr_1=1,
                measure_pixels_overlap_n_px_thr_2=0, count_n_overl_reg_intersection_threshold=None, conv_hull_fract_px_thre_arr_1=3,
                conv_hull_fract_px_thre_arr_2=3, get_conv_hull_fract_arr1_NOpass_arr2_pass_v=0.0, get_conv_hull_fract_arr2_NOpass_v=np.nan):
        
        self.analysis_axis = analysis_axis
        self.roi_structure = roi_structure
        self.shuffle_times = shuffle_times
        self.no_quantification_valu_e = no_quantification_valu_e
        self.channels_binarization_thresholds = channels_binarization_thresholds
        self.transform_to_label_img = transform_to_label_img
        self.get_mask_area_val_4zero_regionprops = get_mask_area_val_4zero_regionprops
        self.count_regions_number_threshold_roi_mask = count_regions_number_threshold_roi_mask
        self.n_of_region_4areas_measure = n_of_region_4areas_measure
        self.reg_eucl_dist_within_arr_val_n_regions_nopass = reg_eucl_dist_within_arr_val_n_regions_nopass
        self.get_convex_hull_min_px_num = get_convex_hull_min_px_num
        self.min_px_over_thresh_common = min_px_over_thresh_common
        self.measure_pixels_overlap_n_px_thr_1 = measure_pixels_overlap_n_px_thr_1
        self.measure_pixels_overlap_n_px_thr_2 = measure_pixels_overlap_n_px_thr_2
        self.count_n_overl_reg_intersection_threshold = count_n_overl_reg_intersection_threshold
        self.conv_hull_fract_px_thre_arr_1 = conv_hull_fract_px_thre_arr_1
        self.conv_hull_fract_px_thre_arr_2 = conv_hull_fract_px_thre_arr_2
        self.get_conv_hull_fract_arr1_NOpass_arr2_pass_v = get_conv_hull_fract_arr1_NOpass_arr2_pass_v
        self.get_conv_hull_fract_arr2_NOpass_v = get_conv_hull_fract_arr2_NOpass_v
    
    def quantify_sample(self, sample_input_folder, roi_maintain=None, roi_exclude=None, roi_3D_maintain=False, roi_3D_exclude=False, roi_position_axis=None):
        """
        files in sample_input_folder must have the same dimensions.
        If roi are provided, for the moment it only works with 2D or 3D arrays as the opening of the rois assumes their positioning in a 2D or 3D array.
        """
        #Form a list with the files in sample_input_folder
        list_of_input_files = listdirNHF(sample_input_folder)

        #Open files in sample_input_folder and collect them in a list - identify the file to use as roi if roi_structure is provided
        collection_of_input_files = []
        collection_of_input_files_names = []
        #Initialize the roi_channel as None
        roi_channel_i = None
        #Iterate through the input files
        for input_f in list_of_input_files:
            #Open the file
            input_file = tifffile.imread(os.path.join(sample_input_folder, input_f))
            #if the opened file name contains roi_structure string
            if self.roi_structure != None:
                #Substitute the opened file to roi_channel_i
                if self.roi_structure in input_f:
                    roi_channel_i = input_file.copy()
                #Otherwise append the opened file to the collection list
                else:
                    #add file in the collection list
                    collection_of_input_files.append(input_file)
                    collection_of_input_files_names.append(input_f)
            #append the opened file to the collection list
            else:
                #add file in the collection list
                collection_of_input_files.append(input_file)
                collection_of_input_files_names.append(input_f)
        
        #Open roi_maintain if it is provided
        if roi_maintain !=None:
            print("this should not be done!")
            #Use the first file of the collection list of files in the input folder as initial reference image
            reference_img_i = collection_of_input_files[0]
            #Get a 2D reference image using analysis_axis if roi_3D is False, else use the entire reference_img_i
            if roi_3D_maintain==False:
                #If the files to analyse are 2D, just use reference_img_i to open the roi file
                if len(reference_img_i.shape)==2:
                    reference_image = reference_img_i
                #If the files to analyse are 3D
                else:
                    #If an axis is provided to position rois (roi_position_axis), use it to get a 2D image out of the reference_img_i
                    if roi_position_axis != None:
                        reference_image = [np.squeeze(a) for a in np.split(reference_img_i, indices_or_sections=reference_img_i.shape[roi_position_axis],axis=roi_position_axis)][0]
                    #If no roi_position_axis but an analysis axis is provided, use the analysis axis to get a 2D image out of the reference_img_i
                    elif self.analysis_axis!=None:
                        reference_image = [np.squeeze(b) for b in np.split(reference_img_i, indices_or_sections=reference_img_i.shape[self.analysis_axis],axis=self.analysis_axis)][0]
                    #If none of the options above are valid, raise a value error
                    else:
                        raise ValueError("can't allocate a 3D roi file in a 3D image if neither roi_position_axis nor analysis_axis are indicated")
                #Open the roi_file and transform it to a numpy array
                roi_2_maintain_i = form_mask_from_roi(roi_maintain,
                                                        reference_img=reference_image,
                                                        ax_position=None,
                                                        return_coordinates=False,
                                                        roi_pixel_value=255,
                                                        background_pixel_val=0,
                                                        output_dtype=np.uint8)
                
                #Match the roi to the dimension of the first file of the collection list of files in the input folder as initial reference image
                roi_2_maintain = match_arrays_dimensions(roi_2_maintain_i, reference_img_i)
            else:
                reference_image = reference_img_i.copy()
                #Use roi_position_axis if provided, else use analysis axis, for allocating roi along their axis
                if roi_position_axis != None:
                    roi_maintain_axis = roi_position_axis
                elif self.analysis_axis!=None:
                    roi_maintain_axis = self.analysis_axis
                else:
                    raise ValueError("can't allocate a 3D roi file in a 3D image if neither roi_position_axis nor analysis_axis are indicated")

                #Open the roi_file and transform it to a numpy array
                roi_2_maintain = form_mask_from_roi(roi_maintain,
                                                    reference_img=reference_image,
                                                    ax_position=roi_maintain_axis,
                                                    return_coordinates=False,
                                                    roi_pixel_value=255,
                                                    background_pixel_val=0,
                                                    output_dtype=np.uint8)
        
        #Associate roi_2_maintain to None if no roi_maintain is None
        else:
            roi_2_maintain = None

        #Open roi_exclude if it is provided
        if roi_exclude !=None:
            print("but there is a roi to exclude...")
            #Use the first file of the collection list of files in the input folder as initial reference image
            reference_img_i_excl = collection_of_input_files[0]
            #Get a 2D reference image using analysis_axis if roi_3D is False, else use the entire reference_img_i_excl
            if roi_3D_exclude==False:

                #If the files to analyse are 2D, just use reference_img_i_excl to open the roi file
                if len(reference_img_i_excl.shape)==2:
                    reference_image_excl = reference_img_i_excl
                #If the files to analyse are 3D
                else:
                    #If an axis is provided to position rois (roi_position_axis), use it to get a 2D image out of the reference_img_i
                    if roi_position_axis != None:
                        reference_image_excl = [np.squeeze(c) for c in np.split(reference_img_i_excl, indices_or_sections=reference_img_i_excl.shape[roi_position_axis],axis=roi_position_axis)][0]
                    #If no roi_position_axis but an analysis axis is provided, use the analysis axis to get a 2D image out of the reference_img_i
                    elif self.analysis_axis!=None:
                        reference_image_excl = [np.squeeze(d) for d in np.split(reference_img_i_excl, indices_or_sections=reference_img_i_excl.shape[self.analysis_axis],axis=self.analysis_axis)][0]
                    #If none of the options above are valid, raise a value error
                    else:
                        raise ValueError("can't allocate a 3D roi file in a 3D image if neither roi_position_axis nor analysis_axis are indicated")
                
                #Open the roi_file and transform it to a numpy array
                roi_2_exclude_i = form_mask_from_roi(roi_exclude,
                                                        reference_img=reference_image_excl,
                                                        ax_position=None,
                                                        return_coordinates=False,
                                                        roi_pixel_value=255,
                                                        background_pixel_val=0,
                                                        output_dtype=np.uint8)
                #Match the roi to the dimension of the first file of the collection list of files in the input folder as initial reference image
                roi_2_exclude = match_arrays_dimensions(roi_2_exclude_i, reference_img_i_excl)
            else:
                reference_image_excl = reference_img_i_excl.copy()

                #Use roi_position_axis if provided, else use analysis axis, for allocating roi along their axis
                if roi_position_axis != None:
                    roi_exclude_axis = roi_position_axis
                elif self.analysis_axis!=None:
                    roi_exclude_axis = self.analysis_axis
                else:
                    raise ValueError("can't allocate a 3D roi file in a 3D image if neither roi_position_axis nor analysis_axis are indicated")
                
                #Open the roi_file and transform it to a numpy array
                roi_2_exclude = form_mask_from_roi(roi_exclude,
                                                    reference_img=reference_image_excl,
                                                    ax_position=roi_exclude_axis,
                                                    return_coordinates=False,
                                                    roi_pixel_value=255,
                                                    background_pixel_val=0,
                                                    output_dtype=np.uint8)
        
        #Associate roi_2_exclude to None, if roi_exclude is None
        else:
            roi_2_exclude = None

        #Combine roi to maintain and roi to exclude in a unique array
        if roi_maintain !=None or roi_exclude !=None:
            #If no roi_structure is provided, initialize and empty array to be used for the combination of the rois
            if self.roi_structure == None:
                zeros_array = np.zeros(collection_of_input_files[0].shape)
                #Combination of rois to maintain and exclude 
                roi_channel = maintain_exclude_image_rois(zeros_array,
                                                             array_mask_to_maintain=roi_2_maintain,
                                                             array_mask_to_exclude=roi_2_exclude,
                                                             thresholds_maintain=0,
                                                             thresholds_exclude=0,
                                                             output_lowval=None,
                                                             output_highval=None,
                                                             output_dtype=np.uint8,
                                                             common_pixels='exclude')
            #If roi_structure is provided, use roi_channel_i for the combination of the rois
            else:
                #Combination of rois to maintain and exclude 
                roi_channel = maintain_exclude_image_rois(roi_channel_i,
                                                             array_mask_to_maintain=roi_2_maintain,
                                                             array_mask_to_exclude=roi_2_exclude,
                                                             thresholds_maintain=0,
                                                             thresholds_exclude=0,
                                                             output_lowval=None,
                                                             output_highval=None,
                                                             output_dtype=np.uint8,
                                                             common_pixels='exclude')
        else:
            roi_channel = roi_channel_i
        
        #Stack all the files in the input directory in a single array on axis 0. Axis 0 becomes my channel_axis
        multi_channel_array = np.stack(collection_of_input_files, axis=0)

        #Define the analysis axis - because structures in the input array are stacked together on axis 0, if analysis axis is provided, it must be increased of 1 unit
        if self.analysis_axis != None:
            quantification_axis = self.analysis_axis+1
        else:
            quantification_axis = self.analysis_axis

        #Quantify channels
        channels_quantifications = quantify_channels(channels_array=multi_channel_array,
                                                     channels_axis=0,
                                                     roi_mask_array=roi_channel,
                                                     analysis_axis=quantification_axis,
                                                     shuffle_times=self.shuffle_times,
                                                     no_quantification_valu_e=self.no_quantification_valu_e,
                                                     channels_binarization_thresholds=self.channels_binarization_thresholds,
                                                     transform_to_label_img=self.transform_to_label_img,
                                                     get_mask_area_val_4zero_regionprops=self.get_mask_area_val_4zero_regionprops,
                                                     count_regions_number_threshold_roi_mask=self.count_regions_number_threshold_roi_mask,
                                                     n_of_region_4areas_measure=self.n_of_region_4areas_measure,
                                                     reg_eucl_dist_within_arr_val_n_regions_nopass=self.reg_eucl_dist_within_arr_val_n_regions_nopass,
                                                     get_convex_hull_min_px_num= self.get_convex_hull_min_px_num,
                                                     min_px_over_thresh_common=self.min_px_over_thresh_common,
                                                     measure_pixels_overlap_n_px_thr_1=self.measure_pixels_overlap_n_px_thr_1,
                                                     measure_pixels_overlap_n_px_thr_2=self.measure_pixels_overlap_n_px_thr_2,
                                                     count_n_overl_reg_intersection_threshold=self.count_n_overl_reg_intersection_threshold,
                                                     conv_hull_fract_px_thre_arr_1=self.conv_hull_fract_px_thre_arr_1,
                                                     conv_hull_fract_px_thre_arr_2=self.conv_hull_fract_px_thre_arr_2,
                                                     get_conv_hull_fract_arr1_NOpass_arr2_pass_v=self.get_conv_hull_fract_arr1_NOpass_arr2_pass_v,
                                                     get_conv_hull_fract_arr2_NOpass_v=self.get_conv_hull_fract_arr2_NOpass_v)

        return channels_quantifications, collection_of_input_files_names, multi_channel_array
    
    def change_columns_names(self, channels_new_names, channels_quantifications_df, collection_of_input_files_names, iteration_axis=None, new_name_iteration_axis=None,
                             return_column_names_map_dict=False):
        if iteration_axis!=None:
            if new_name_iteration_axis==None:
                print("WARNING! it is indicated an iteration axis but no new name is provided, the name will be maintained")

        #Copy channels_quantifications_df
        channels_quantifications_df_copy = channels_quantifications_df.copy()

        #Initialize a dictionary to map channels to their new name
        channels_names_mapping_dict = {}

        #Iterate through the list of files. Note: the list of files contains the files in order as they are opened by the listdirNHF function in utils.py
        #Because this function is used in quantify_sample (above) this is the order of the files passed to quantify_channels (within quantify_samples). Thus
        #The order of the channels numbering corresponds to the order of the files in the list collection_of_input_files_names
        for c, f in enumerate(collection_of_input_files_names):
            #Iterate through the list of channels_new_names
            for c_n_n in channels_new_names:
                #If the string c_n_n is in the name of the file in collection_of_input_files_names
                if c_n_n in f:
                    #Re-form the string of the channel as it is saved in the columns name of the channels_quantifications_df dataframe (refer to quantify_channels in multi_channels_multi_quantifications.py)
                    channel_initial_name = 'ch_'+str(c)
                    #Link the channel initial string to the c_n_n string in channels_names_mapping_dict
                    channels_names_mapping_dict[channel_initial_name]=c_n_n

        #Get columns names
        channels_quantifications_df_columns = channels_quantifications_df_copy.columns

        #Initialize a dictionary to map name changes
        new_column_names_map = {}

        #Iterate through the dictionary mapping channels to their new names
        for ch in channels_names_mapping_dict:

            #Iterate through the columns names
            for clm in channels_quantifications_df_columns:

                #If an iteration axis is provided, check if the column name fits that of the iteration axis
                if clm == 'axis_'+str(iteration_axis+1): #Note, the +1 compensates for the channels which are created in position 0 when forming the channel array to analyse in quantify_sample (see above)
                    #Check if a new name is given for the iteration axis
                    if new_name_iteration_axis!=None:
                        
                        new_column_names_map[clm]=new_name_iteration_axis
                    #If no new name is given, just report the old name
                    else:
                        new_column_names_map[clm]=clm

                #If the channel name is in the column name
                if ch in clm:
                    
                    #Because some column names have multi channels (compatative measurements), firs check if the column name had alredy been changed for a channel
                    if clm in new_column_names_map:

                        #If the column name had already been changed for a channel, replace the new channel in the already changed name
                        column_new_name = new_column_names_map[clm].replace(ch, channels_names_mapping_dict[ch])

                    #If it is the first time that the column name is changed
                    else:

                        #Replace the channel string in the initial column name
                        column_new_name = clm.replace(ch, channels_names_mapping_dict[ch])

                    #Update the dictionary mapping channels to their new names
                    new_column_names_map[clm]=column_new_name
                
                #If the channel name is not in the dictionary
                else:
                    #Add the column name, without modification, only if it is the first iteration through the column (to avoid overwriting of changes made in former iterations)
                    if clm not in new_column_names_map:
                        new_column_names_map[clm]=clm

        
        #Substitute the columns names in the channels_quantifications_df_copy with the new column names
        new_col_channels_quantifications_df = channels_quantifications_df_copy.rename(columns=new_column_names_map)

        if return_column_names_map_dict:
            return new_col_channels_quantifications_df, new_column_names_map
        
        else:
            return new_col_channels_quantifications_df

    def get_roi_files(self, root_roi_folder, sample_name_2_match):
        
        #Form the directory for the sample_name_2_match in root_roi_folder
        sample_roi_directory = os.path.join(root_roi_folder, sample_name_2_match)

        #If the directory exist
        if os.path.exists(sample_roi_directory):

            #List the files inside the directory - NOTE: it is expected a single file!
            roi_file_list = listdirNHF(sample_roi_directory)

            #Print a warning if no file is present in the folder
            if len(roi_file_list)==0:
                print(f"WARNING! For sample {sample_name_2_match} no file is present in either roi_maintain or roi_exclude")
                return None
            
            #If a file is present, return a full directory to the file
            else:
                return os.path.join(sample_roi_directory,roi_file_list[0])

        #Return None if the directory does not exist
        else:
            return None


    def batch_quantification(self, root_input_folder, root_output_folder, channels_new__names,
                             root_folder_roi=None, new_name_iteration__axis=None, roi_3D__maintain=False,
                             roi_3D__exclude=False, roi_position__axis=None):
        
        #Create a list of the folders in root_input_folder
        list_of_samples = listdirNHF(root_input_folder)

        #Iterate through the folders in list_of_samples
        for f in list_of_samples:
            print("---", f)
            #Join folder to the root directory
            f_directory = os.path.join(root_input_folder, f)

            #Create the directory a matching folder in root_output_folder
            f_output_folder = os.path.join(root_output_folder, f)

            #Create output_folder, if it does not exist
            if not os.path.exists(f_output_folder):
                os.makedirs(f_output_folder)
            
            #Fish the ROIs files for the sample at f, if root_folder_roi exists
            if root_folder_roi!=None:
                roi_maintain_for_f = self.get_roi_files(os.path.join(root_folder_roi, "roi_maintain"), f)
                roi_exclude_for_f = self.get_roi_files(os.path.join(root_folder_roi, "roi_exclude"), f)
            else:
                #Link roi_maintain_for_f and roi_exclude_for_f to None
                roi_maintain_for_f = None
                roi_exclude_for_f = None
            
            #Get the result measurements and the order of the analysed samples for the sample f
            f_result_measurements, f_samples_order, f_multi_channel_array = self.quantify_sample(f_directory,
                                                                                                 roi_maintain=roi_maintain_for_f,
                                                                                                 roi_exclude=roi_exclude_for_f,
                                                                                                 roi_3D_maintain=roi_3D__maintain,
                                                                                                 roi_3D_exclude=roi_3D__exclude,
                                                                                                 roi_position_axis=roi_position__axis)
            #Change the names of the columns in the quantification dataframe
            f_results_measurements_new_col = self.change_columns_names(channels_new_names=channels_new__names,
                                                                       channels_quantifications_df=f_result_measurements,
                                                                       collection_of_input_files_names=f_samples_order,
                                                                       iteration_axis=self.analysis_axis,
                                                                       new_name_iteration_axis=new_name_iteration__axis,
                                                                       return_column_names_map_dict=False)
            
            #Form the saving name of the output dataframe
            output_df_saving_name = f+"_raw_measurements.csv"

            #Save the result in the output folder
            f_results_measurements_new_col.to_csv(os.path.join(f_output_folder, output_df_saving_name), index=False)
        
        print("FINISHED!")
        
