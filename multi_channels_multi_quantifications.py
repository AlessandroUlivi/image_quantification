import numpy as np
import pandas as pd
from co_localization_measurements import measure_pixels_overlap, measure_regions_euclidean_distances, count_number_of_overlapping_regions
from counting_measurements import count_regions_number
from geometric_measurements import get_mask_area, get_areas_of_regions_in_mask
from topological_measurement import get_convex_hull_fractions



# get_mask_area
# count_regions_number
# get_areas_of_regions_in_mask
# measure_pixels_overlap,
# measure_regions_euclidean_distances,
# count_number_of_overlapping_regions
# get_convex_hull_fractions


def quantify_channels(channels_array, channels_axis=0, roi_mask_array=None, analysis_axis=None, shuffle_times=0, add_means_stdv=False, roi_mask_analysis_axis=None,
                      channels_binarization_thresholds=None):
    """
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
                roi_mask_analysis_axis_2use = analysis_axis
            else:
                roi_mask_analysis_axis_2use = roi_mask_analysis_axis
            roi_mask_list = [np.squeeze(d) for d in np.split(roi_mask_array_copy, indices_or_sections=roi_mask_array_copy.shape[roi_mask_analysis_axis_2use], axis=roi_mask_analysis_axis_2use)]
    
    #Set binarization thresholds to 0 for all channels, if channels channels_binarization_thresholds is not provided. Use provided values othewise.
    if channels_binarization_thresholds==None:
        ch_bin_thresh_2use = [0 for bin_t in range(channels_array_copy.shape[channels_axis])]
    else:
        ch_bin_thresh_2use = channels_binarization_thresholds

    #Initialize a dictionary to be used to be used to form the output datafram
    measurements_dict = {}

    #If analysis axis is provided:
    if analysis_axis != None:
        # print("==="*10)
        # print("---analyze a specific axis---")
        # print("==="*10)
        # Iterate through the analysis axis
        for ixd, idx_array in enumerate([np.squeeze(a) for a in np.split(channels_array_copy,
                                                                         indices_or_sections=channels_array_copy.shape[analysis_axis],
                                                                         axis=analysis_axis)]):
            # print("---", ixd, idx_array.shape)
            # Iterate through the channels
            for ch_n, ch_array in enumerate([np.squeeze(b) for b in np.split(idx_array,
                                                                             indices_or_sections=idx_array.shape[channels_axis_2use],
                                                                             axis=channels_axis_2use)]):
                # print("ch ", ch_n, ch_array.shape)
                #Get the region_to_quantify, if it is provided
                if hasattr(roi_mask_array, "__len__"):
                    ch_n_roi_mask_array = roi_mask_list[ch_n]
                    print("shape roi mask", ch_n_roi_mask_array.shape)
                else:
                    ch_n_roi_mask_array=roi_mask_array_copy #which should be meaning None

                #Get mask area
                ch_n_area = get_mask_area(ch_array, roi_mas_k=ch_n_roi_mask_array, binarization_threshold=ch_bin_thresh_2use[ch_n])


    #If the analysis axis is not provided          
    else:
        print("==="*10)
        print("---NO analysis of a specific axis---")
        print("==="*10)

        # Iterate through the channels
        for ch_n1, ch_array1 in enumerate([np.squeeze(c) for c in np.split(channels_array_copy,
                                                                             indices_or_sections=channels_array_copy.shape[channels_axis_2use],
                                                                             axis=channels_axis_2use)]):
            # print("ch ", ch_n1, ch_array1.shape)
            if hasattr(roi_mask_array, "__len__"):
                print("shape roi mask", roi_mask_array_copy.shape)
            else:
                print(roi_mask_array_copy)
    return
