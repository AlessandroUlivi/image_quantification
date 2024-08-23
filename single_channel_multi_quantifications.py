import numpy as np
import pandas as pd
from scipy.stats import sem
from co_localization_measurements import measure_pixels_overlap, measure_regions_euclidean_distances, count_number_of_overlapping_regions
from counting_measurements import count_regions_number
from geometric_measurements import get_mask_area, get_areas_of_regions_in_mask, get_covex_hull_from_mask
from topological_measurements import get_convex_hull_fraction, measure_regions_euclidean_distances_within_array
from utils import match_arrays_dimensions


# measure_pixels_overlap,
# measure_regions_euclidean_distances,
# count_number_of_overlapping_regions
# get_convex_hull_fractions
# measure_regions_euclidean_distances_within_array


def quantify_single_channel(channels_array, roi_mask_array=None, analysis_axis=None, no_quantification_valu_e=np.nan,
                      channels_binarization_thresholds=0, transform_to_label_img=False, get_mask_area_val_4zero_regionprops=0,
                      count_regions_number_threshold_roi_mask=0, n_of_region_4areas_measure=0, reg_eucl_dist_within_arr_val_n_regions_nopass=1,
                      get_convex_hull_min_px_num=2):
    """
    The full documentation is in file single_channel_multi_quantifications_documentation.rtf.
    """
    #=========================================
    #=========  SUPPORTING FUNCTIONS =========
    def modify_dictionary(result_valu_e, dict2modify, root_key_name, channel_1_number=None, channel_2_number=None):

        if isinstance(result_valu_e, list) or isinstance(result_valu_e, tuple):
            if len(result_valu_e)==0:
                print("WARNING! a result_valu_e has been passed to modify_dictionary, it is a list or tuple, but the length is 0")
            for i,v in enumerate(result_valu_e):
                if channel_1_number!=None:
                    if channel_2_number!=None:
                        result_name = f"{root_key_name}_ch_{channel_1_number}_ch_{channel_2_number}_iter_{i}"
                    else:
                        result_name = f"{root_key_name}_ch_{channel_1_number}_iter_{i}"
                    
                    if result_name not in dict2modify:
                        dict2modify[result_name]=[v]
                    else:
                        dict2modify[result_name].append(v)
                    # print(result_name, len(dict2modify[result_name]))
                else:
                    result_name = f"{root_key_name}_iter_{i}"

                    if result_name not in dict2modify:
                        dict2modify[result_name]=[v]
                    else:
                        dict2modify[result_name].append(v)
                    # print(result_name, len(dict2modify[result_name]))
        else:
            if channel_1_number!=None:
                if channel_2_number!=None:
                    result_name = f"{root_key_name}_ch_{channel_1_number}_ch_{channel_2_number}"
                else:
                    result_name = f"{root_key_name}_ch_{channel_1_number}"
                
                if result_name not in dict2modify:
                    dict2modify[result_name]=[result_valu_e]
                else:
                    dict2modify[result_name].append(result_valu_e)
                # print(result_name, len(dict2modify[result_name]))
            else:
                result_name=root_key_name

                if result_name not in dict2modify:
                    dict2modify[result_name]=[result_valu_e]
                else:
                    dict2modify[result_name].append(result_valu_e)
                # print(result_name, len(dict2modify[result_name]))

    def get_mean_median_std_sem_min_max_results(results_measurements, no_quantification_value=np.nan):
        """
        many of the output measurements are lists and the length of the list depends/corresponds to the number of quantified elements (e.g. regions).
        - if the list has 3 or more quantifications. Return the mean, median, standard deviation, standard error of mean, max and min values of the measurement list.
        - if the list has 2 quantifications. Return the mean, median, max and min values of the measurement list and return no_quantification_value for the standard deviation and standard
        error of means.
        - if the list has 1 quantification. Return the value for the mean, median, max and min and return no_quantification_value for the standard deviation and standard
        error of means.
        - the list has 0 quantifications. Return no_quantification_value for mean, median, standard deviation, standard error of mean, max and min.

        Inputs:
        - results_measurements. list or tuple.
        - no_quantification_value. any. Default np.nan.

        Output:
        tuple. Pos-0, number of quantified elements. Pos-1, mean of results_measurements. Pos-2, median of results_measurements. Pos-3, standard deviation of results_measurements.
        Pos-4, standard error of the means of results_measurements. Pos-5, min of results_measurements. Pos-6, max of results_measurements. 
        """
        if not isinstance(results_measurements,list) and not isinstance(results_measurements, tuple):
            print("results_measurements must be either a list or a tuple")
            return
        
        if len(results_measurements)>2:
            mean_results_measurements = np.mean(results_measurements)
            median_results_measurements = np.median(results_measurements)
            stdv_results_measurements = np.std(results_measurements)
            sem_results_measurements = sem(results_measurements)
            min_results_measurements = np.min(results_measurements)
            max_results_measurements = np.max(results_measurements)
        elif len(results_measurements)==2:
            mean_results_measurements = np.mean(results_measurements)
            median_results_measurements = np.median(results_measurements)
            stdv_results_measurements = no_quantification_value
            sem_results_measurements = no_quantification_value
            min_results_measurements = np.min(results_measurements)
            max_results_measurements = np.max(results_measurements)
        elif len(results_measurements)==1:
            mean_results_measurements = results_measurements[0]
            median_results_measurements = results_measurements[0]
            stdv_results_measurements = no_quantification_value
            sem_results_measurements = no_quantification_value
            min_results_measurements = results_measurements[0]
            max_results_measurements = results_measurements[0]
        else:
            mean_results_measurements = no_quantification_value
            median_results_measurements = no_quantification_value
            stdv_results_measurements = no_quantification_value
            sem_results_measurements = no_quantification_value
            min_results_measurements = no_quantification_value
            max_results_measurements = no_quantification_value
        return len(results_measurements), mean_results_measurements, median_results_measurements, stdv_results_measurements, sem_results_measurements, min_results_measurements, max_results_measurements
    
    def set_thresholds_2use(input_thresholds, channels_stac_k):
        """
        Inputs:
        - input_thresholds. None, int/float, tuple/list or ndarray.
        - channels_stac_k. ndarray.

        Outputs:
        - if input_thresholds is int/float. The output is an ndarray of the same shape of channels_stac_k and all values set to input_threshols.
        - if input_thresholds is tuple/list. The output is an ndarray of shape=channels_stac_k.shape+1. The extra dimension is in position -1 and its size corresponds to the
        length of input_thresholds. Each ndarray of shape=channels_stac_k.shape which is stacked along the -1 dimension, contains one of the values of input_thresholds.
        - if input_thresholds is ndarray. The output is input_thresholds. NOTE: if multiple thresholds are reported and input_thresholds is ndarray, the multiple threshold values
        must be indicated on axis -1.
        """
        if isinstance(input_thresholds, int) or isinstance(input_thresholds, float) or isinstance(input_thresholds, bool):
            return np.where(np.zeros(channels_stac_k.shape)==0, input_thresholds,input_thresholds)
        elif isinstance(input_thresholds, tuple) or isinstance(input_thresholds, tuple):
            return np.stack([np.where(np.zeros(channels_stac_k.shape)==0, input_thresholds[k],input_thresholds[k]) for k in range(len(input_thresholds))], axis=-1)
        else:
            return input_thresholds

    def split_thresholds_arrays(thr_array, split_axis, multi_thresholds=False):
        """
        Splits thr_array along the axis split_axis. If multi_thresholds array is True, split_axis is reduced of 1 number.

        Inputs:
        - thr_array. ndarray.
        - split_axis. int. Must be <len(thr_array.shape). The dimension, among those present in thr_array, on which to split thr_array.
        - multi_thehresholds. Bool. Defauls False. If True, reduced split_axis of 1 before thr_array is splat. 

        This function works in the context of set_thresholds_2use and quantify_channels. For certain quantification functions it is required to indicate multiple thresholds. These
        thresholds are assumed to be indicated in the -1 axis of the threshold array which is either the output of set_threshold_2use or directly input to quantify_channels. For this
        reason, when multiple thresholds are reported, the threshold array has an additional axis than channel_array (see quantify_channels inputs), in position -1.
        When channels_array and threshold arrays have to be split, the axis along which to do the split is referring to channel array and does not take into consideration the
        extra dimension of thresholds array. This results in a wrong indexing when the index is indicated using negative numbers. The present function compensates for this fact.
        """
        if multi_thresholds:
            if split_axis<0:
                split_axis_2use = split_axis-1
            else:
                split_axis_2use = split_axis
        else:
            split_axis_2use = split_axis
        
        return [np.squeeze(z) for z in np.split(thr_array, indices_or_sections=thr_array.shape[split_axis_2use], axis=split_axis_2use)]
    
    def get_threshold_from_list(array_threshold, multi_value_array=False, multi_value_axis=-1, get_a_single_value=True):
        """
        Given:
        array_threshold. ndarray.

        The function returns:
        - if multi_value_array==False (default).
            - if get_a_single_value==True. The function returns the average value of array_threshold, as an int or float.
            - if get_a_single_value==False. The function returns array_threshold.
        - if multi_value_array==True. The function splits array_threshold in sub-arrays along multi_value_axis (defaul is -1).
            - if get_a_single_value==True. The function returns a tuple with the average value of each sub-array of array_threshold along the multi_value_axis.
            - if get_a_single_value==False. The function return the sub-arrays obtained by splitting array_threshold along the multi_value_axis. The output dtype is a list.
        
        This function works in the context of set_thresholds_2use and quantify_channels. For certain quantification functions it is required to indicate multiple thresholds. These
        thresholds must be indicated in the -1 axis of the threshold array which is either the output of set_threshold_2use or directly input to quantify_channels. For this reason,
        when multiple thresholds are reported, the threshold array has an additional axis than channel_array (see quantify_channels inputs), in position -1. When the thresholds have
        to be retrieved, this function allows to retrieve each individual multi-threshold from the extra dimension, if required.
        NOTE:
        - although it is possible to change the axis where multi-thresholds are reported by changing multi_value_axis, the whole processing was conceptualize for having these
        values in the axis -1 and it hasn't been tested for different situations.
        - Although it is possible to retrieve an entire array instead of a single threshold value, the process is conceptualized for retrieving a single value and it hasn't been
        tested for different situations.
        """
        if multi_value_array:
            multi_thresholds_split = [np.squeeze(y) for y in np.split(array_threshold, indices_or_sections=array_threshold.shape[multi_value_axis], axis=multi_value_axis)]
            if get_a_single_value:
                return tuple([np.mean(mta) for mta in multi_thresholds_split])
            else:
                return multi_thresholds_split
        else:
            if get_a_single_value:
                return np.mean(array_threshold)
            else:
                return array_threshold

    #=======================================
    #=========  PREPARE INPUT DATA =========
    #Copy input arrays
    channels_array_copy = channels_array.copy()

    #If roi_mask_array is provided make sure it is has the same shape of channels_array, else, copy the None
    if hasattr(roi_mask_array, "__len__"):
        roi_mask_array_copy_i = roi_mask_array.copy()
        roi_mask_array_2use = match_arrays_dimensions(roi_mask_array_copy_i, channels_array_copy)
    else:
        roi_mask_array_2use = roi_mask_array

    #=========================================================
    #=========  GET THRESHOLDS IN THE CORRECT FORMAT =========
    #Set binarization thresholds to 0 for all channels, if channels_binarization_thresholds is not provided. Use provided values othewise.
    ch_bin_thresh_2use = set_thresholds_2use(channels_binarization_thresholds, channels_stac_k=channels_array_copy)

    #Set to False transform_to_label_img if transform_to_label_img is not provided. Use the provided value otherwise
    transform_to_label_img_2use = set_thresholds_2use(transform_to_label_img, channels_stac_k=channels_array_copy)
    
    #Set val_4zero_regionprops in get_mask_area as 0 by defaut, if None is provided as input. Use the provided value otherwise.
    val_4zero_regionprops_2use = set_thresholds_2use(get_mask_area_val_4zero_regionprops, channels_stac_k=channels_array_copy)

    #Set threshold_roi_mask in count_regions_number as 0 by defaut, if None is provided as input. Use the provided value otherwise.
    threshold_roi_mask_2use = set_thresholds_2use(count_regions_number_threshold_roi_mask, channels_stac_k=channels_array_copy)

    #Set to 0 the highpass threshold for calculating the mean, median, min and max area of regions within a channel, if None is provided as input. Use the provided value otherwise.
    n_of_region_4areas_measure_2use = set_thresholds_2use(n_of_region_4areas_measure, channels_stac_k=channels_array_copy)

    #Set to 1 the highpass threshold for calculating distances within regions of a channel, if reg_eucl_dist_within_arr_val_n_regions_nopass is not provided.
    #Use the provided value otherwise
    reg_eucl_dist_within_arr_val_n_regions_nopass_2use = set_thresholds_2use(reg_eucl_dist_within_arr_val_n_regions_nopass, channels_stac_k=channels_array_copy)

    #Set to 2 the highpass threshold for calculating the convex hull of a channel, if get_convex_hull_min_px_num is not provided.
    #Use the provided value otherwise
    get_convex_hull_min_px_num_2use = set_thresholds_2use(get_convex_hull_min_px_num, channels_stac_k=channels_array_copy)

    #==========================================
    #=========  INITIALIZE THE OUTPUT =========
    #Initialize a dictionary to be used to form the output datafram
    measurements_dict = {}
    
    #==================================================
    #=========  USE ANALYSIS AXIS IF PROVIDED =========
    #If analysis axis is provided:
    if analysis_axis != None:
        
        #==============================================================================
        #=========  PREPARE ROI AND THRESHOLDS FOR ITERATION ON ANALYSIS AXIS =========
        #Split the roi_mask_array on the analysis_axis, if roi_mask_array is provided
        if hasattr(roi_mask_array, "__len__"):
            roi_mask_array_2use_1 = [np.squeeze(w) for w in np.split(roi_mask_array_2use,
                                                                         indices_or_sections=channels_array_copy.shape[analysis_axis],
                                                                         axis=analysis_axis)]
            # print("roi_mask after analysis axis split: ", len(roi_mask_array_2use_1), roi_mask_array_2use_1[0].shape)
        else:
            roi_mask_array_2use_1 = roi_mask_array_2use #which should be meaning None

        #Split threshold arrays on the analysis_axis
        ch_bin_thresh_2use_1 = split_thresholds_arrays(ch_bin_thresh_2use, split_axis=analysis_axis, multi_thresholds=False)
        transform_to_label_img_2use_1 = split_thresholds_arrays(transform_to_label_img_2use, split_axis=analysis_axis, multi_thresholds=False)
        val_4zero_regionprops_2use_1 = split_thresholds_arrays(val_4zero_regionprops_2use, split_axis=analysis_axis, multi_thresholds=False)
        threshold_roi_mask_2use_1 = split_thresholds_arrays(threshold_roi_mask_2use, split_axis=analysis_axis, multi_thresholds=False)
        n_of_region_4areas_measure_2use_1 = split_thresholds_arrays(n_of_region_4areas_measure_2use, split_axis=analysis_axis, multi_thresholds=False)
        reg_eucl_dist_within_arr_val_n_regions_nopass_2use_1 = split_thresholds_arrays(reg_eucl_dist_within_arr_val_n_regions_nopass_2use, split_axis=analysis_axis, multi_thresholds=False)
        get_convex_hull_min_px_num_2use_1 = split_thresholds_arrays(get_convex_hull_min_px_num_2use, split_axis=analysis_axis, multi_thresholds=False)

        #=================================================
        #=========  ITERATE ON THE ANALYSIS AXIS =========
        # Iterate through the analysis axis
        for ixd, idx_array in enumerate([np.squeeze(a) for a in np.split(channels_array_copy,
                                                                         indices_or_sections=channels_array_copy.shape[analysis_axis],
                                                                         axis=analysis_axis)]):
            # print("==="*3, ixd)
            # print(idx_array.shape)
            #============================================
            #========= UPDATE OUTPUT DICTIONARY =========
            #Update measurements_dict, which will be used to form the output dataframe
            modify_dictionary(result_valu_e=ixd, dict2modify=measurements_dict, root_key_name='axis_'+str(analysis_axis), channel_1_number=None, channel_2_number=None)
            
            #==================================================
            #=========  GET ROI IN THE CORRECT FORMAT =========
            #Get the roi_mask for channel ch_n, if it is provided
            if hasattr(roi_mask_array, "__len__"):
                ch_n_roi_mask_array = roi_mask_array_2use_1[ixd]
                # print("final shape roi mask", ch_n_roi_mask_array.shape)
            else:
                ch_n_roi_mask_array= roi_mask_array_2use_1 #which should be meaning None
            # print("final roi mask ", ch_n_roi_mask_array.shape)
            #=================================
            #=========  ANALYSE AREA =========
            #Get threshold value for channel ch_n and index ixd in the analysis axis
            ch_n_ixd_binarization_threshold = get_threshold_from_list(ch_bin_thresh_2use_1[ixd],
                                                                            multi_value_array=False,
                                                                            multi_value_axis=-1,
                                                                            get_a_single_value=True)
            #Get threshold value for channel ch_n and index ixd in the analysis axis
            ch_n_ixd_value_4_zero_regionprops = get_threshold_from_list(val_4zero_regionprops_2use_1[ixd],
                                                                            multi_value_array=False,
                                                                            multi_value_axis=-1,
                                                                            get_a_single_value=True)
            #Measure area as number of pixels
            ch_n_area_px, ch_n_area_props = get_mask_area(idx_array,
                                                              roi_mas_k=ch_n_roi_mask_array,
                                                              binarization_threshold=ch_n_ixd_binarization_threshold,
                                                              value_4_zero_regionprops=ch_n_ixd_value_4_zero_regionprops)
                
            #============================================
            #========= UPDATE OUTPUT DICTIONARY =========
            #Update measurements_dict, which will be used to form the output dataframe
            modify_dictionary(result_valu_e=ch_n_area_px, dict2modify=measurements_dict, root_key_name='area', channel_1_number=0, channel_2_number=None)

            #==================================
            #=========  COUNT REGIONS =========
            #Get threshold value for channel ch_n and index ixd in the analysis axis
            ch_n_ixd_threshold_roi_mask = get_threshold_from_list(threshold_roi_mask_2use_1[ixd],
                                                                            multi_value_array=False,
                                                                            multi_value_axis=-1,
                                                                            get_a_single_value=True)
            #Count region number
            ch_n_regions_number = count_regions_number(idx_array,
                                                           roi_mask=ch_n_roi_mask_array,
                                                           threshold_input_arr=ch_n_ixd_binarization_threshold,
                                                           threshold_roi_mask=ch_n_ixd_threshold_roi_mask)
            # print("n of regions ", ch_n_regions_number)
            #============================================
            #========= UPDATE OUTPUT DICTIONARY =========
            #Update measurements_dict, which will be used to form the output dataframe
            modify_dictionary(result_valu_e=ch_n_regions_number, dict2modify=measurements_dict, root_key_name='region_number', channel_1_number=0, channel_2_number=None)

            #===========================================
            #=========  MEASURE REGIONS' AREAS =========
            #Get threshold value for channel ch_n and index ixd in the analysis axis
            ch_n_ixd_n_of_region_4areas_measure = get_threshold_from_list(n_of_region_4areas_measure_2use_1[ixd],
                                                                                multi_value_array=False,
                                                                                multi_value_axis=-1,
                                                                                get_a_single_value=True)
            #Get threshold value for channel ch_n at index ixd in the analysis axis
            ch_n_ixd_transform_to_label_img = get_threshold_from_list(transform_to_label_img_2use_1[ixd],
                                                                            multi_value_array=False,
                                                                            multi_value_axis=-1,
                                                                            get_a_single_value=True)
                
            #Calculate the area of each individual region in ch_n, if there are >n_of_region_4areas_measure_2use regions. Alternatively,
            # link area measurements to NaN values
            if ch_n_regions_number>ch_n_ixd_n_of_region_4areas_measure:
                #Get the areas of the regions within the channel
                ch_n_regions_areas = get_areas_of_regions_in_mask(idx_array,
                                                                        roi__mask=ch_n_roi_mask_array,
                                                                        transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                        binarization_threshold=ch_n_ixd_binarization_threshold)
                    
                #ch_n_regions_areas is a list with the areas of each region in ch_n
                #Get mean, median, stdv, sem, max and min regions' area. Get no_quantification_valu_e values if a minimum number of areas is not detected
                ch_n_regions_areas_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n_regions_areas, no_quantification_value=no_quantification_valu_e)
                ch_n_regions_mean_area = ch_n_regions_areas_results[1]
                ch_n_regions_median_area = ch_n_regions_areas_results[2]
                ch_n_regions_stdv_area = ch_n_regions_areas_results[3]
                ch_n_regions_sem_area = ch_n_regions_areas_results[4]
                ch_n_regions_min_area = ch_n_regions_areas_results[5]
                ch_n_regions_max_area = ch_n_regions_areas_results[6]
            else:
                ch_n_regions_mean_area = np.nan
                ch_n_regions_median_area = np.nan
                ch_n_regions_stdv_area = np.nan
                ch_n_regions_sem_area = np.nan
                ch_n_regions_min_area = np.nan
                ch_n_regions_max_area = np.nan
                
            #============================================
            #========= UPDATE OUTPUT DICTIONARY =========
            #Update measurements_dict, which will be used to form the output dataframe
            modify_dictionary(result_valu_e=ch_n_regions_mean_area, dict2modify=measurements_dict, root_key_name='mean_regions_area', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=ch_n_regions_median_area, dict2modify=measurements_dict, root_key_name='median_regions_area', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=ch_n_regions_stdv_area, dict2modify=measurements_dict, root_key_name='stdv_regions_area', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=ch_n_regions_sem_area, dict2modify=measurements_dict, root_key_name='sem_regions_area', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=ch_n_regions_min_area, dict2modify=measurements_dict, root_key_name='min_regions_area', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=ch_n_regions_max_area, dict2modify=measurements_dict, root_key_name='max_regions_area', channel_1_number=0, channel_2_number=None)

            #=======================================================================
            #=========  MEASURE INTER-REGIONS DISTANCES WITHIN THE CHANNEL =========
            #Get threshold value for channel ch_n and index ixd in the analysis axis
            ch_n_ixd_highpass_n_regions_4distance = get_threshold_from_list(reg_eucl_dist_within_arr_val_n_regions_nopass_2use_1[ixd],
                                                                            multi_value_array=False,
                                                                            multi_value_axis=-1,
                                                                            get_a_single_value=True)
            #Measure region minimum distances
            ch_n_regions_min_distances, ch_n_regions_min_dict = measure_regions_euclidean_distances_within_array(idx_array,
                                                                                                                        roi__mask=ch_n_roi_mask_array,
                                                                                                                        desired__distance='min',
                                                                                                                        highpass_n_regions=ch_n_ixd_highpass_n_regions_4distance,
                                                                                                                        transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                                                                        label_img_thres=ch_n_ixd_binarization_threshold,
                                                                                                                        return_excluded_distances=False,
                                                                                                                        val_n_regions_nopass=np.nan)
                
            #ch_n_regions_min_distances is a list with the min distance of each region in ch_n to the rest of the regions of ch_n. If less or equal to
            # ch_n_ixd_highpass_n_regions_4distance are present the output is np.nan.
            #Get mean, median, stdv, sem, max and min regions' min distances. Get no_quantification_valu_e values if a minimum number of regions is not detected
            if isinstance(ch_n_regions_min_distances,list):
                ch_n_regions_min_distances_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n_regions_min_distances, no_quantification_value=no_quantification_valu_e)
                num_ch_n_regions_min_distances = ch_n_regions_min_distances_results[0]
                mean_ch_n_regions_min_distances = ch_n_regions_min_distances_results[1]
                median_ch_n_regions_min_distances = ch_n_regions_min_distances_results[2]
                std_ch_n_regions_min_distances = ch_n_regions_min_distances_results[3]
                sem_ch_n_regions_min_distances = ch_n_regions_min_distances_results[4]
                min_ch_n_regions_min_distances = ch_n_regions_min_distances_results[5]
                max_ch_n_regions_min_distances = ch_n_regions_min_distances_results[6]

            else:
                num_ch_n_regions_min_distances = np.nan
                mean_ch_n_regions_min_distances = np.nan
                median_ch_n_regions_min_distances = np.nan
                std_ch_n_regions_min_distances = np.nan
                sem_ch_n_regions_min_distances = np.nan
                min_ch_n_regions_min_distances = np.nan
                max_ch_n_regions_min_distances = np.nan
                
            #Measure region maximum distances
            ch_n_regions_max_distances, ch_n_regions_max_dict = measure_regions_euclidean_distances_within_array(idx_array,
                                                                                                                        roi__mask=ch_n_roi_mask_array,
                                                                                                                        desired__distance='max',
                                                                                                                        highpass_n_regions=ch_n_ixd_highpass_n_regions_4distance,
                                                                                                                        transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                                                                        label_img_thres=ch_n_ixd_binarization_threshold,
                                                                                                                        return_excluded_distances=False,
                                                                                                                        val_n_regions_nopass=np.nan)
                
            #ch_n_regions_max_distances is a list with the max distance of each region in ch_n to the rest of the regions of ch_n. If less or equal to
            # ch_n_ixd_highpass_n_regions_4distance are present the output is np.nan.
            #Get mean, median, stdv, sem, max and mim regions' max distances. Get no_quantification_valu_e values if a minimum number of regions is not detected
            if isinstance(ch_n_regions_max_distances,list):
                ch_n_regions_max_distances_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n_regions_max_distances, no_quantification_value=no_quantification_valu_e)
                num_ch_n_regions_max_distances = ch_n_regions_max_distances_results[0]
                mean_ch_n_regions_max_distances = ch_n_regions_max_distances_results[1]
                median_ch_n_regions_max_distances = ch_n_regions_max_distances_results[2]
                std_ch_n_regions_max_distances = ch_n_regions_max_distances_results[3]
                sem_ch_n_regions_max_distances = ch_n_regions_max_distances_results[4]
                min_ch_n_regions_max_distances = ch_n_regions_max_distances_results[5]
                max_ch_n_regions_max_distances = ch_n_regions_max_distances_results[6]
            else:
                num_ch_n_regions_max_distances = np.nan
                mean_ch_n_regions_max_distances = np.nan
                median_ch_n_regions_max_distances = np.nan
                std_ch_n_regions_max_distances = np.nan
                sem_ch_n_regions_max_distances = np.nan
                min_ch_n_regions_max_distances = np.nan
                max_ch_n_regions_max_distances = np.nan

            #Measure region mean distances
            ch_n_regions_mean_distances, ch_n_regions_mean_dict = measure_regions_euclidean_distances_within_array(idx_array,
                                                                                                                        roi__mask=ch_n_roi_mask_array,
                                                                                                                        desired__distance='mean',
                                                                                                                        highpass_n_regions=ch_n_ixd_highpass_n_regions_4distance,
                                                                                                                        transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                                                                        label_img_thres=ch_n_ixd_binarization_threshold,
                                                                                                                        return_excluded_distances=False,
                                                                                                                        val_n_regions_nopass=np.nan)
                
            #ch_n_regions_mean_distances is a list with the mean distance of each region in ch_n to the rest of the regions of ch_n. If less or equal to
            # ch_n_ixd_highpass_n_regions_4distance are present the output is np.nan.
            #Get mean, median, stdv, sem, max and mim regions' mean distances. Get no_quantification_valu_e values if a minimum number of regions is not detected
            if isinstance(ch_n_regions_mean_distances,list):
                ch_n_regions_mean_distances_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n_regions_mean_distances, no_quantification_value=no_quantification_valu_e)
                num_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[0]
                mean_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[1]
                median_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[2]
                std_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[3]
                sem_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[4]
                min_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[5]
                max_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[6]
            else:
                num_ch_n_regions_mean_distances = np.nan
                mean_ch_n_regions_mean_distances = np.nan
                median_ch_n_regions_mean_distances = np.nan
                std_ch_n_regions_mean_distances = np.nan
                sem_ch_n_regions_mean_distances = np.nan
                min_ch_n_regions_mean_distances = np.nan
                max_ch_n_regions_mean_distances = np.nan

            #============================================
            #========= UPDATE OUTPUT DICTIONARY =========
            #Update measurements_dict, which will be used to form the output dataframe
            modify_dictionary(result_valu_e=num_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='number_region_min_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=mean_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='mean_region_min_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=median_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='median_region_min_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=std_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='stdv_region_min_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=sem_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='sem_region_min_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=min_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='min_region_min_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=max_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='max_region_min_distances', channel_1_number=0, channel_2_number=None)

            modify_dictionary(result_valu_e=num_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='number_region_max_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=mean_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='mean_region_max_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=median_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='median_region_max_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=std_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='stdv_region_max_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=sem_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='sem_region_max_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=min_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='min_region_max_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=max_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='max_region_max_distances', channel_1_number=0, channel_2_number=None)

            modify_dictionary(result_valu_e=num_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='number_region_mean_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=mean_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='mean_region_mean_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=median_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='median_region_mean_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=std_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='stdv_region_mean_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=sem_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='sem_region_mean_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=min_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='min_region_mean_distances', channel_1_number=0, channel_2_number=None)
            modify_dictionary(result_valu_e=max_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='max_region_mean_distances', channel_1_number=0, channel_2_number=None)
             
                
            #=============================================
            #=========  MEASURE CONVEX HULL AREA =========
            ch_n_min_px_num = get_threshold_from_list(get_convex_hull_min_px_num_2use_1[ixd],
                                                            multi_value_array=False,
                                                            multi_value_axis=-1,
                                                            get_a_single_value=True)
            #Get the convex hull
            ch_n_convex_hull, ch_n_convex_hull_coords = get_covex_hull_from_mask(idx_array,
                                                                                     roi_mask=ch_n_roi_mask_array,
                                                                                     threshold_4arr=ch_n_ixd_binarization_threshold,
                                                                                     threshold_4roi=ch_n_ixd_threshold_roi_mask,
                                                                                     min_px_num=ch_n_min_px_num,
                                                                                     value_4no_quantification=None)
            #The result of get_covex_hull_from_mask is the whole convex hull. The area/volume have to be extracted
            if ch_n_convex_hull !=None:
                ch_n_convex_hull_volume = ch_n_convex_hull.volume
            else:
                ch_n_convex_hull_volume = no_quantification_valu_e
                
            #============================================
            #========= UPDATE OUTPUT DICTIONARY =========
            modify_dictionary(result_valu_e=ch_n_convex_hull_volume, dict2modify=measurements_dict, root_key_name='convex_hull_volume', channel_1_number=0, channel_2_number=None)


    #If the analysis axis is not provided - NOTE: all the analyses are repeated identical without iteration on the analysis axis
    else:

        #=================================
        #=========  ANALYSE AREA =========
        # print(ch_bin_thresh_2use_2[ch_n].shape)
        # print(val_4zero_regionprops_2use_2[ch_n].shape)
        #Get threshold value for channel ch_n
        ch_n_ixd_binarization_threshold = get_threshold_from_list(ch_bin_thresh_2use,
                                                                        multi_value_array=False,
                                                                        multi_value_axis=-1,
                                                                        get_a_single_value=True)
        #Get threshold value for channel ch_n
        ch_n_ixd_value_4_zero_regionprops = get_threshold_from_list(val_4zero_regionprops_2use,
                                                                        multi_value_array=False,
                                                                        multi_value_axis=-1,
                                                                        get_a_single_value=True)
        #Get the area as number of pixels
        ch_n_area_px, ch_n_area_props = get_mask_area(channels_array_copy,
                                                            roi_mas_k=roi_mask_array_2use,
                                                            binarization_threshold=ch_n_ixd_binarization_threshold,
                                                            value_4_zero_regionprops=ch_n_ixd_value_4_zero_regionprops)

        #============================================
        #========= UPDATE OUTPUT DICTIONARY =========
        #Update measurements_dict, which will be used to form the output dataframe
        modify_dictionary(result_valu_e=ch_n_area_px, dict2modify=measurements_dict, root_key_name='area', channel_1_number=0, channel_2_number=None)

        #==================================
        #=========  COUNT REGIONS =========
        #Get threshold value for channel ch_n
        ch_n_ixd_threshold_roi_mask = get_threshold_from_list(threshold_roi_mask_2use,
                                                                        multi_value_array=False,
                                                                        multi_value_axis=-1,
                                                                        get_a_single_value=True)
        #Count region number
        ch_n_regions_number = count_regions_number(channels_array_copy,
                                                        roi_mask=roi_mask_array_2use,
                                                        threshold_input_arr=ch_n_ixd_binarization_threshold,
                                                        threshold_roi_mask=ch_n_ixd_threshold_roi_mask)
        # print("n of regions ", ch_n_regions_number)
        #============================================
        #========= UPDATE OUTPUT DICTIONARY =========
        #Update measurements_dict, which will be used to form the output dataframe
        modify_dictionary(result_valu_e=ch_n_regions_number, dict2modify=measurements_dict, root_key_name='region_number', channel_1_number=0, channel_2_number=None)

        #===========================================
        #=========  MEASURE REGIONS' AREAS =========
        #Get threshold value for channel ch_n
        ch_n_ixd_n_of_region_4areas_measure = get_threshold_from_list(n_of_region_4areas_measure_2use,
                                                                            multi_value_array=False,
                                                                            multi_value_axis=-1,
                                                                            get_a_single_value=True)
        #Get threshold value for channel ch_n
        ch_n_ixd_transform_to_label_img = get_threshold_from_list(transform_to_label_img_2use,
                                                                        multi_value_array=False,
                                                                        multi_value_axis=-1,
                                                                        get_a_single_value=True)
                
        #Calculate the area of each individual region in ch_n, if there are >n_of_region_4areas_measure_2use regions. Alternatively,
        # link area measurements to NaN values
        if ch_n_regions_number>ch_n_ixd_n_of_region_4areas_measure:
            #Get the areas of the regions within the channel
            ch_n_regions_areas = get_areas_of_regions_in_mask(channels_array_copy,
                                                                    roi__mask=roi_mask_array_2use,
                                                                    transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                    binarization_threshold=ch_n_ixd_binarization_threshold)
                    
            #ch_n_regions_areas is a list with the areas of each region in ch_n
            #Get mean, median, stdv, sem, max and min regions' area. Get no_quantification_valu_e values if a minimum number of areas is not detected
            ch_n_regions_areas_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n_regions_areas, no_quantification_value=no_quantification_valu_e)
            ch_n_regions_mean_area = ch_n_regions_areas_results[1]
            ch_n_regions_median_area = ch_n_regions_areas_results[2]
            ch_n_regions_stdv_area = ch_n_regions_areas_results[3]
            ch_n_regions_sem_area = ch_n_regions_areas_results[4]
            ch_n_regions_min_area = ch_n_regions_areas_results[5]
            ch_n_regions_max_area = ch_n_regions_areas_results[6]
        else:
            ch_n_regions_mean_area = np.nan
            ch_n_regions_median_area = np.nan
            ch_n_regions_stdv_area = np.nan
            ch_n_regions_sem_area = np.nan
            ch_n_regions_min_area = np.nan
            ch_n_regions_max_area = np.nan
                
        #============================================
        #========= UPDATE OUTPUT DICTIONARY =========
        #Update measurements_dict, which will be used to form the output dataframe
        modify_dictionary(result_valu_e=ch_n_regions_mean_area, dict2modify=measurements_dict, root_key_name='mean_regions_area', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=ch_n_regions_median_area, dict2modify=measurements_dict, root_key_name='median_regions_area', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=ch_n_regions_stdv_area, dict2modify=measurements_dict, root_key_name='stdv_regions_area', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=ch_n_regions_sem_area, dict2modify=measurements_dict, root_key_name='sem_regions_area', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=ch_n_regions_min_area, dict2modify=measurements_dict, root_key_name='min_regions_area', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=ch_n_regions_max_area, dict2modify=measurements_dict, root_key_name='max_regions_area', channel_1_number=0, channel_2_number=None)

        #=======================================================================
        #=========  MEASURE INTER-REGIONS DISTANCES WITHIN THE CHANNEL =========
        #Get threshold value for channel ch_n
        ch_n_ixd_highpass_n_regions_4distance = get_threshold_from_list(reg_eucl_dist_within_arr_val_n_regions_nopass_2use,
                                                                            multi_value_array=False,
                                                                            multi_value_axis=-1,
                                                                            get_a_single_value=True)
        #Measure region minimum distances
        ch_n_regions_min_distances, ch_n_regions_min_dict = measure_regions_euclidean_distances_within_array(channels_array_copy,
                                                                                                                    roi__mask=roi_mask_array_2use,
                                                                                                                    desired__distance='min',
                                                                                                                    highpass_n_regions=ch_n_ixd_highpass_n_regions_4distance,
                                                                                                                    transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                                                                    label_img_thres=ch_n_ixd_binarization_threshold,
                                                                                                                    return_excluded_distances=False,
                                                                                                                    val_n_regions_nopass=np.nan)
                
        #ch_n_regions_min_distances is a list with the min distance of each region in ch_n to the rest of the regions of ch_n. If less or equal to
        # ch_n_ixd_highpass_n_regions_4distance are present the output is np.nan.
        #Get mean, median, stdv, sem, max and min regions' min distances. Get no_quantification_valu_e values if a minimum number of regions is not detected
        if isinstance(ch_n_regions_min_distances,list):
            ch_n_regions_min_distances_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n_regions_min_distances, no_quantification_value=no_quantification_valu_e)
            num_ch_n_regions_min_distances = ch_n_regions_min_distances_results[0]
            mean_ch_n_regions_min_distances = ch_n_regions_min_distances_results[1]
            median_ch_n_regions_min_distances = ch_n_regions_min_distances_results[2]
            std_ch_n_regions_min_distances = ch_n_regions_min_distances_results[3]
            sem_ch_n_regions_min_distances = ch_n_regions_min_distances_results[4]
            min_ch_n_regions_min_distances = ch_n_regions_min_distances_results[5]
            max_ch_n_regions_min_distances = ch_n_regions_min_distances_results[6]

        else:
            num_ch_n_regions_min_distances = np.nan
            mean_ch_n_regions_min_distances = np.nan
            median_ch_n_regions_min_distances = np.nan
            std_ch_n_regions_min_distances = np.nan
            sem_ch_n_regions_min_distances = np.nan
            min_ch_n_regions_min_distances = np.nan
            max_ch_n_regions_min_distances = np.nan
                
        #Measure region maximum distances
        ch_n_regions_max_distances, ch_n_regions_max_dict = measure_regions_euclidean_distances_within_array(channels_array_copy,
                                                                                                                    roi__mask=roi_mask_array_2use,
                                                                                                                    desired__distance='max',
                                                                                                                    highpass_n_regions=ch_n_ixd_highpass_n_regions_4distance,
                                                                                                                    transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                                                                    label_img_thres=ch_n_ixd_binarization_threshold,
                                                                                                                    return_excluded_distances=False,
                                                                                                                    val_n_regions_nopass=np.nan)
                
        #ch_n_regions_max_distances is a list with the max distance of each region in ch_n to the rest of the regions of ch_n. If less or equal to
        # ch_n_ixd_highpass_n_regions_4distance are present the output is np.nan.
        #Get mean, median, stdv, sem, max and mim regions' max distances. Get no_quantification_valu_e values if a minimum number of regions is not detected
        if isinstance(ch_n_regions_max_distances,list):
            ch_n_regions_max_distances_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n_regions_max_distances, no_quantification_value=no_quantification_valu_e)
            num_ch_n_regions_max_distances = ch_n_regions_max_distances_results[0]
            mean_ch_n_regions_max_distances = ch_n_regions_max_distances_results[1]
            median_ch_n_regions_max_distances = ch_n_regions_max_distances_results[2]
            std_ch_n_regions_max_distances = ch_n_regions_max_distances_results[3]
            sem_ch_n_regions_max_distances = ch_n_regions_max_distances_results[4]
            min_ch_n_regions_max_distances = ch_n_regions_max_distances_results[5]
            max_ch_n_regions_max_distances = ch_n_regions_max_distances_results[6]
        else:
            num_ch_n_regions_max_distances = np.nan
            mean_ch_n_regions_max_distances = np.nan
            median_ch_n_regions_max_distances = np.nan
            std_ch_n_regions_max_distances = np.nan
            sem_ch_n_regions_max_distances = np.nan
            min_ch_n_regions_max_distances = np.nan
            max_ch_n_regions_max_distances = np.nan

        #Measure region mean distances
        ch_n_regions_mean_distances, ch_n_regions_mean_dict = measure_regions_euclidean_distances_within_array(channels_array_copy,
                                                                                                                    roi__mask=roi_mask_array_2use,
                                                                                                                    desired__distance='mean',
                                                                                                                    highpass_n_regions=ch_n_ixd_highpass_n_regions_4distance,
                                                                                                                    transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                                                                    label_img_thres=ch_n_ixd_binarization_threshold,
                                                                                                                    return_excluded_distances=False,
                                                                                                                    val_n_regions_nopass=np.nan)
                
        #ch_n_regions_mean_distances is a list with the mean distance of each region in ch_n to the rest of the regions of ch_n. If less or equal to
        # ch_n_ixd_highpass_n_regions_4distance are present the output is np.nan.
        #Get mean, median, stdv, sem, max and mim regions' mean distances. Get no_quantification_valu_e values if a minimum number of regions is not detected
        if isinstance(ch_n_regions_mean_distances,list):
            ch_n_regions_mean_distances_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n_regions_mean_distances, no_quantification_value=no_quantification_valu_e)
            num_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[0]
            mean_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[1]
            median_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[2]
            std_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[3]
            sem_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[4]
            min_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[5]
            max_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[6]
        else:
            num_ch_n_regions_mean_distances = np.nan
            mean_ch_n_regions_mean_distances = np.nan
            median_ch_n_regions_mean_distances = np.nan
            std_ch_n_regions_mean_distances = np.nan
            sem_ch_n_regions_mean_distances = np.nan
            min_ch_n_regions_mean_distances = np.nan
            max_ch_n_regions_mean_distances = np.nan

        #============================================
        #========= UPDATE OUTPUT DICTIONARY =========
        #Update measurements_dict, which will be used to form the output dataframe
        modify_dictionary(result_valu_e=num_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='number_region_min_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=mean_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='mean_region_min_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=median_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='median_region_min_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=std_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='stdv_region_min_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=sem_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='sem_region_min_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=min_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='min_region_min_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=max_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='max_region_min_distances', channel_1_number=0, channel_2_number=None)

        modify_dictionary(result_valu_e=num_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='number_region_max_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=mean_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='mean_region_max_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=median_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='median_region_max_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=std_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='stdv_region_max_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=sem_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='sem_region_max_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=min_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='min_region_max_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=max_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='max_region_max_distances', channel_1_number=0, channel_2_number=None)

        modify_dictionary(result_valu_e=num_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='number_region_mean_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=mean_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='mean_region_mean_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=median_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='median_region_mean_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=std_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='stdv_region_mean_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=sem_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='sem_region_mean_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=min_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='min_region_mean_distances', channel_1_number=0, channel_2_number=None)
        modify_dictionary(result_valu_e=max_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='max_region_mean_distances', channel_1_number=0, channel_2_number=None)

        #=============================================
        #=========  MEASURE CONVEX HULL AREA =========
        ch_n_min_px_num = get_threshold_from_list(get_convex_hull_min_px_num_2use,
                                                        multi_value_array=False,
                                                        multi_value_axis=-1,
                                                        get_a_single_value=True)
        #Get the convex hull
        ch_n_convex_hull, ch_n_convex_hull_coords = get_covex_hull_from_mask(channels_array_copy,
                                                                                    roi_mask=roi_mask_array_2use,
                                                                                    threshold_4arr=ch_n_ixd_binarization_threshold,
                                                                                    threshold_4roi=ch_n_ixd_threshold_roi_mask,
                                                                                    min_px_num=ch_n_min_px_num,
                                                                                    value_4no_quantification=None)
        #The result of get_covex_hull_from_mask is the whole convex hull. The area/volume have to be extracted
        if ch_n_convex_hull !=None:
            ch_n_convex_hull_volume = ch_n_convex_hull.volume
        else:
            ch_n_convex_hull_volume = no_quantification_valu_e
                
        #============================================
        #========= UPDATE OUTPUT DICTIONARY =========
        modify_dictionary(result_valu_e=ch_n_convex_hull_volume, dict2modify=measurements_dict, root_key_name='convex_hull_volume', channel_1_number=0, channel_2_number=None)

    #Use measurements_dict to form the output dataframe
    output_dataframe = pd.DataFrame.from_dict(measurements_dict)
    return output_dataframe
