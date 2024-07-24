import numpy as np
import pandas as pd
from co_localization_measurements import measure_pixels_overlap, measure_regions_euclidean_distances, count_number_of_overlapping_regions
from counting_measurements import count_regions_number
from geometric_measurements import get_mask_area, get_areas_of_regions_in_mask
from topological_measurement import get_convex_hull_fraction

# measure_pixels_overlap,
# measure_regions_euclidean_distances,
# count_number_of_overlapping_regions
# get_convex_hull_fractions



def quantify_channels(channels_array, channels_axis=0, roi_mask_array=None, analysis_axis=None, shuffle_times=0, add_means_stdv=False, roi_mask_analysis_axis=None,
                      channels_binarization_thresholds=0, get_mask_area_val_4zero_regionprops=0, count_regions_number_threshold_roi_mask=0, n_of_region_4areas_measure=0,
                      min_px_over_thresh_common=(0,0)):
    """
    seems to work with 3 limitations: 1) when providing the thresholds they array cannot contain axis of size 1. 2) when channel_axis/analysis_axis is in position 0 it can't
    be indicated using negative number indexing (for example -10). 3) If a custom array of thresholds is provided and multiple thresholds are required (e.g. for comparison
    functions, the multiple thresholds must be in position -1)
    - roi_mask_array is the same for all the channels. It has shape channels_array-1.
    - min_px_over_thresh_common. the number of o pixels both channels must pass to continue with paired measurements.
    """

    def modify_dictionary(dict2modify, key_name, valu_e=None, interations_times=1):
        for i in range(interations_times):
            if interations_times==1:
                key_name_1 = key_name
            else:
                key_name_1 = key_name + "_"+str(i)
            
            if key_name_1 not in dict2modify:
                if valu_e == None:
                    dict2modify[key_name_1]=[]
                else:
                    dict2modify[key_name_1]=[valu_e]
            else:
                if valu_e != None:
                    dict2modify[key_name_1].append(valu_e)
                else:
                    print("WARNING! ", key_name_1, "'s value is updated by no value is provided for the update. None will be used instead")
                    dict2modify[key_name_1].append(valu_e)
    
    # def set_thresholds_2use(input_thresholds, default_value, range_2use):
    #     if input_thresholds==None:
    #         return [default_value for th in range(range_2use)]
    #     elif isinstance(input_thresholds, int) or isinstance(input_thresholds, float) or isinstance(input_thresholds, tuple):
    #         return [input_thresholds for th1 in range(range_2use)]
    #     else:
    #         return input_thresholds
    def set_thresholds_2use(input_thresholds, channels_stac_k):
        """
        Inputs:
        - input_thresholds. None, int/float, tuple/list or ndarray.
        - channels_stac_k. ndarray.

        Outputs:
        - if input_thresholds is int/float. The output is an ndarray of the same shape of channels_stac_k and all values set to input_threshols.
        - if input_thresholds is tuple/list. The output is an ndarray of shape=channels_stac_k.shape+1. The extra dimension is in position -1 and its size corresponds to the
        length of input_thresholds. Each ndarray of shape=channels_stac_k.shape which is stacked along the -1 dimension, contains one of the values of input_thresholds.
        - if input_thresholds is ndarray. The output is input_thresholds.
        """
        if isinstance(input_thresholds, int) or isinstance(input_thresholds, float):
            return np.where(np.zeros(channels_stac_k.shape)==0, input_thresholds,input_thresholds)
        elif isinstance(input_thresholds, tuple) or isinstance(input_thresholds, tuple):
            return np.stack([np.where(np.zeros(channels_stac_k.shape)==0, input_thresholds[k],input_thresholds[k]) for k in range(len(input_thresholds))], axis=-1)
        else:
            return input_thresholds

    def split_thresholds_arrays(thr_array, split_axis, multi_thresholds=False):
        """
        Splits thr_array along split_axis. If multi_thresholds array is True, split_axis is reduced of 1 number.
        This function works in the context of 
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
        The whole quantify_channels function is conceptualized so that array_threshold should have the same number of dimensions of the array to threshold, apart if multiple
        thresholds are required. If multiple thresholds are required they are passed to input_thresholds_2 as a tuple or a list and array_threshold would have the same number
        of the array to threshold, Plus an extra dimension, at position -1, having size equal to the number of multi-thresholds.
        Thus if input_thresholds_2 is a tuple or a list, the function first splits indivual arrays along the axis multi_value_axis (default -1). Secondly get the thresholds associated
        to each of these individual arrays. It returns such thresholds in a list.
        If input_thresholds_2 is not a tuple or a list, 
        """
        if multi_value_array:
            multi_thresholds_split = [np.squeeze(y) for y in np.split()]
        return

    #Make sure that channels_axis and analysis axis are not the same axis
    assert channels_axis != analysis_axis, "channels_axis can't be the same of analysis_axis"

    #If arrays are split along an axis to analyses, because channels_axis will work after the splitting, recude it of 1 unit to compensate for the recuded dimension
    if analysis_axis != None:
        if channels_axis>analysis_axis:
            channels_axis_2use = channels_axis-1
        else:
            channels_axis_2use = channels_axis
    else:
        channels_axis_2use = channels_axis

    #Copy input arrays
    channels_array_copy = channels_array.copy()
    roi_mask_array_copy = roi_mask_array.copy()

    #If roi_mask_array is provided and analysis_axis is provided, transform roi_mask_array_copy into a list of sub-arrays along the axis to analyse
    if hasattr(roi_mask_array, "__len__"):
        if analysis_axis != None:

            #Use analysis_axis as the axis along which split the sub-arrays if no specific roi_mask_analysis_axis is provided. Use the provided axis, otherwise
            if roi_mask_analysis_axis == None:
                if channels_axis>analysis_axis:
                    roi_mask_analysis_axis_2use = analysis_axis
                else:
                    roi_mask_analysis_axis_2use = analysis_axis-1
            else:
                #Make sure that the number of roi_mask ndarrays along roi_mask_analysis_axis matches the number of ndarrays along the 
                assert len(roi_mask_list)==channels_array_copy.shape[analysis_axis], "the size of roi_mask_analysis_axis must match the size of analysis_axis"
                roi_mask_analysis_axis_2use = roi_mask_analysis_axis
            roi_mask_list = [np.squeeze(d) for d in np.split(roi_mask_array_copy, indices_or_sections=roi_mask_array_copy.shape[roi_mask_analysis_axis_2use], axis=roi_mask_analysis_axis_2use)]
            # print("length of roi_Mask list ", len(roi_mask_list))

    # #Set binarization thresholds to 0 for all channels, if channels channels_binarization_thresholds is not provided. Use provided values othewise.
    # ch_bin_thresh_2use = set_thresholds_2use(channels_binarization_thresholds, default_value=0, range_2use=channels_array_copy.shape[channels_axis])
    
    # #Set val_4zero_regionprops in get_mask_area as 0 by defaut, if None is provided as input. Use the provided value otherwise.
    # val_4zero_regionprops_2use = set_thresholds_2use(get_mask_area_val_4zero_regionprops, default_value=0, range_2use=channels_array_copy.shape[channels_axis])

    # #Set threshold_roi_mask in count_regions_number as 0 by defaut, if None is provided as input. Use the provided value otherwise.
    # threshold_roi_mask_2use = set_thresholds_2use(count_regions_number_threshold_roi_mask, default_value=0, range_2use=channels_array_copy.shape[channels_axis])

    # #Set to 0 the highpass threshold for calculating the mean, median, min and max area of regions within a channel, if None is provided as input. Use the provided value otherwise.
    # n_of_region_4areas_measure_2use = set_thresholds_2use(n_of_region_4areas_measure, default_value=0, range_2use=channels_array_copy.shape[channels_axis])

    # #Set to 0 and 0 the min number of pixels for proceeding with measurements, if min_px_over_thresh_common is not provided. Use the provided thresholds otherwise
    # min_px_over_thresh_common_2use = set_thresholds_2use(min_px_over_thresh_common, default_value=(0,0), range_2use=channels_array_copy.shape[channels_axis])



    #Set binarization thresholds to 0 for all channels, if channels channels_binarization_thresholds is not provided. Use provided values othewise.
    ch_bin_thresh_2use = set_thresholds_2use(channels_binarization_thresholds, channels_stac_k=channels_array_copy)
    
    #Set val_4zero_regionprops in get_mask_area as 0 by defaut, if None is provided as input. Use the provided value otherwise.
    val_4zero_regionprops_2use = set_thresholds_2use(get_mask_area_val_4zero_regionprops, channels_stac_k=channels_array_copy)

    #Set threshold_roi_mask in count_regions_number as 0 by defaut, if None is provided as input. Use the provided value otherwise.
    threshold_roi_mask_2use = set_thresholds_2use(count_regions_number_threshold_roi_mask, channels_stac_k=channels_array_copy)

    #Set to 0 the highpass threshold for calculating the mean, median, min and max area of regions within a channel, if None is provided as input. Use the provided value otherwise.
    n_of_region_4areas_measure_2use = set_thresholds_2use(n_of_region_4areas_measure, channels_stac_k=channels_array_copy)

    #Set to 0 and 0 the min number of pixels for proceeding with measurements, if min_px_over_thresh_common is not provided. Use the provided thresholds otherwise
    min_px_over_thresh_common_2use = set_thresholds_2use(min_px_over_thresh_common, channels_stac_k=channels_array_copy)

    print("===== initial")
    print(ch_bin_thresh_2use.shape)
    print(val_4zero_regionprops_2use.shape)
    print(threshold_roi_mask_2use.shape)
    print(n_of_region_4areas_measure_2use.shape)
    print(min_px_over_thresh_common_2use.shape)

    #Initialize a dictionary to be used to be used to form the output datafram
    measurements_dict = {}

    #If analysis axis is provided:
    if analysis_axis != None:
        # print("==="*10)
        # print("---analyze a specific axis---")
        # print("==="*10)

        #Split threshold arrays on the analysis_axis
        ch_bin_thresh_2use_1 = split_thresholds_arrays(ch_bin_thresh_2use, split_axis=analysis_axis, multi_thresholds=False)
        val_4zero_regionprops_2use_1 = split_thresholds_arrays(val_4zero_regionprops_2use, split_axis=analysis_axis, multi_thresholds=False)
        threshold_roi_mask_2use_1 = split_thresholds_arrays(threshold_roi_mask_2use, split_axis=analysis_axis, multi_thresholds=False)
        n_of_region_4areas_measure_2use_1 = split_thresholds_arrays(n_of_region_4areas_measure_2use, split_axis=analysis_axis, multi_thresholds=False)
        min_px_over_thresh_common_2use_1 = split_thresholds_arrays( min_px_over_thresh_common_2use, split_axis=analysis_axis, multi_thresholds=True)
        print("===== analysis axis split")
        print(len(ch_bin_thresh_2use_1),ch_bin_thresh_2use_1[0].shape)
        print(len(val_4zero_regionprops_2use_1),val_4zero_regionprops_2use_1[0].shape)
        print(len(threshold_roi_mask_2use_1),threshold_roi_mask_2use_1[0].shape)
        print(len(n_of_region_4areas_measure_2use_1),n_of_region_4areas_measure_2use_1[0].shape)
        print(len(min_px_over_thresh_common_2use_1),min_px_over_thresh_common_2use_1[0].shape)

        # Iterate through the analysis axis
        for ixd, idx_array in enumerate([np.squeeze(a) for a in np.split(channels_array_copy,
                                                                         indices_or_sections=channels_array_copy.shape[analysis_axis],
                                                                         axis=analysis_axis)]):
            if ixd<4:
                print(ixd, idx_array.shape)

            #Get the individual channels array as a list
            ch_arrays_list = [np.squeeze(b) for b in np.split(idx_array, indices_or_sections=idx_array.shape[channels_axis_2use], axis=channels_axis_2use)]

            #Also split on the channel axis the thresholds' arrays corresponding to the ixd-th index along the analysis_axis
            ch_bin_thresh_2use_2 = split_thresholds_arrays(ch_bin_thresh_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            val_4zero_regionprops_2use_2 = split_thresholds_arrays(val_4zero_regionprops_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            threshold_roi_mask_2use_2 = split_thresholds_arrays(threshold_roi_mask_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            n_of_region_4areas_measure_2use_2 = split_thresholds_arrays(n_of_region_4areas_measure_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            min_px_over_thresh_common_2use_2 = split_thresholds_arrays( min_px_over_thresh_common_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=True)

            print("===== channels axis split")
            print("input ", len(ch_arrays_list),ch_arrays_list[0].shape)
            print(len(ch_bin_thresh_2use_2),ch_bin_thresh_2use_2[0].shape)
            print(len(val_4zero_regionprops_2use_2),val_4zero_regionprops_2use_2[0].shape)
            print(len(threshold_roi_mask_2use_2),threshold_roi_mask_2use_2[0].shape)
            print(len(n_of_region_4areas_measure_2use_2),n_of_region_4areas_measure_2use_2[0].shape)
            print(len(min_px_over_thresh_common_2use_2),min_px_over_thresh_common_2use_2[0].shape)


            # Iterate through the channels
            for ch_n, ch_array in enumerate(ch_arrays_list):
                if ch_n<4:
                    print("ch ", ch_n, ch_array.shape)

                #Get the region_to_quantify, if it is provided
                if hasattr(roi_mask_array, "__len__"):
                    ch_n_roi_mask_array = roi_mask_list[ixd]
                    print("shape roi mask", ch_n_roi_mask_array.shape)
                else:
                    ch_n_roi_mask_array=roi_mask_array_copy #which should be meaning None

                #Get mask area
                #Get threshold value for channel ch_n and index ixd in the analysis axis
                ch_n_ixd_binarization_threshold = ch_bin_thresh_2use_2[ch_n] #=====HERE! this will be an array. I have to define a function which gets the correct value. It should not be required a function, however it should take into account if input_threshold was a tuple/list...
                
                # ch_n_area_px, ch_n_area_props = get_mask_area(ch_array,
                #                                               roi_mas_k=ch_n_roi_mask_array,
                #                                               binarization_threshold=ch_bin_thresh_2use[ch_n],
                #                                               value_4_zero_regionprops=val_4zero_regionprops_2use[ch_n])
                
    #             #Count region number
    #             ch_n_regions_number = count_regions_number(ch_array,
    #                                                        roi_mask=ch_n_roi_mask_array,
    #                                                        threshold_input_arr=ch_bin_thresh_2use[ch_n],
    #                                                        threshold_roi_mask=threshold_roi_mask_2use[ch_n])
                
    #             #Calculate mean, median, max and min regions' area, if there are >n_of_region_4areas_measure_2use regions. Alternatively,
    #             # link mean, median, max and min variables to NaN values
    #             if ch_n_regions_number>n_of_region_4areas_measure_2use[ch_n]:
    #                 #Get the areas of the regions within the channel
    #                 ch_n_regions_areas = get_areas_of_regions_in_mask(ch_array,
    #                                                                 roi__mask=ch_n_roi_mask_array,
    #                                                                 transform_to_label_img=True,
    #                                                                 binarization_threshold=ch_bin_thresh_2use[ch_n])
                    
    #                 #Get mean, median, max and min regions' area. Get NaN values if a minimum number of areas is not detected
    #                 ch_n_regions_mean_area = np.mean(ch_n_regions_areas)
    #                 ch_n_regions_median_area = np.median(ch_n_regions_areas)
    #                 ch_n_regions_max_area = np.amax(ch_n_regions_areas)
    #                 ch_n_regions_min_area = np.amin(ch_n_regions_areas)
    #             else:
    #                 ch_n_regions_mean_area = np.nan
    #                 ch_n_regions_median_area = np.nan
    #                 ch_n_regions_max_area = np.nan
    #                 ch_n_regions_min_area = np.nan

    #             #Iterate trough the channels a second time, to get the relative measurements
    #             for cchh_nn, cchh_nn_array in enumerate(ch_arrays_list):
                    
    #                 #Avoid measuring the channel angainst itself
    #                 if ch_n != cchh_nn:
                        
    #                     #Count he number of pixels in the second channel
    #                     cchh_nn_area_px, cchh_nn_area_props = get_mask_area(cchh_nn_array,
    #                                                                         roi_mas_k=ch_n_roi_mask_array,
    #                                                                         binarization_threshold=ch_bin_thresh_2use[cchh_nn],
    #                                                                         value_4_zero_regionprops=val_4zero_regionprops_2use[cchh_nn])

    #                     #Do the following analyses only if both channels pass 

    # #                     #Measure pixels' overlap
    # #                     ch_n__cchh_nn_overlap_i = measure_pixels_overlap(ch_array,
    # #                                                                    cchh_nn_array,
    # #                                                                    roi_mask=ch_n_roi_mask_array,
    # #                                                                    shuffle_times=shuffle_times,
    # #                                                                    n_px_thr_1=min_px_over_thresh_2use[ch_n][0],
    # #                                                                    n_px_thr_2=min_px_over_thresh_2use[ch_n][1],
    # #                                                                    val_threshold_arr_1=ch_bin_thresh_2use[ch_n],
    # #                                                                    val_threshold_arr_2=ch_bin_thresh_2use[cchh_nn])
    # #                     if isinstance(ch_n__cchh_nn_overlap_i, tuple):
    # #                         ch_n__cchh_nn_overlap = ch_n__cchh_nn_overlap_i[0]
    # #                         ch_n__cchh_nn_overlap_shuff = ch_n__cchh_nn_overlap_i[1]
    # #                     else:
    # #                         ch_n__cchh_nn_overlap = np.NaN
    # #                         ch_n__cchh_nn_overlap_shuff = [np.NaN for shf in range(shuffle_times)]



    # # #If the analysis axis is not provided          
    # # else:
    # #     print("==="*10)
    # #     print("---NO analysis of a specific axis---")
    # #     print("==="*10)

    # #     # Iterate through the channels
    # #     for ch_n1, ch_array1 in enumerate([np.squeeze(c) for c in np.split(channels_array_copy,
    # #                                                                          indices_or_sections=channels_array_copy.shape[channels_axis_2use],
    # #                                                                          axis=channels_axis_2use)]):
    # #         if hasattr(roi_mask_array, "__len__"):
    # #             print("shape roi mask", roi_mask_array_copy.shape)
    # #         else:
    # #             print(roi_mask_array_copy)
    # return
