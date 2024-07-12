import numpy as np
from scipy.spatial import ConvexHull
from skimage.measure import label, regionprops


def get_mask_area(input_array, roi_mas_k=None, binarization_threshold=0):
    """
    Returns the area of the region of interest in a binary mask both in terms of number of pixels of the mask and in terms of number of pixels scaled by pixel-area.

    Inputs:
    - input_array. ndarray. Binary array. Pixels of interest are assumed to be the pixels whose value is higher than binarization_threshold (default 0).
    - roi_mas_k. Optional. ndarray of the same size of input_array. Binary mask. If provided, restricts the analysis to a region of interest. The region of interest is assumed to
    the positive pixels in roi_mas_k.
    - binarization_threshold. Int or float. Default 0. Defines the highpass threshold to distinguish pixels of interest from background in input_array. Pixels whose value
    is >binarization_threshold are considered pixels of interest. The rest of the pixels are considered background.

    Outputs: tuple.
    - position-0. The area of the region of interest in input_array as total number of its pixels (the pixels of interest).
    - position-1. The area of the region of interest in input_array as number of pixels scaled by pixel-area. Refer to
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops for further documentation.
    """
    #Copy input_array, binarize it using binarization_threshold and set the values to 1 and 0, where 1 are assumed to be the pixels of interest
    input_array_01 = np.where(input_array>binarization_threshold, 1, 0)
    
    #If roi_mas_k is provided, copy it and use it to set to 0 pixels in input_array corresponded by background values in roi_mas_k
    if hasattr(roi_mas_k, "__len__"):
        roi_mas_k_copy = roi_mas_k.copy()
        array_to_process = np.where(roi_mas_k_copy>0, input_array_01, 0)
    else:
        array_to_process = input_array_01

    #Get the area of the input_array region of interest as number of pixels of interest
    area_as_poi_number = np.sum(array_to_process)

    #Use regionprops to calculate the area of the binary mask
    array_to_process_props = regionprops(array_to_process)[0]
    
    #Get the area of the input_array region of interest number of pixels scaled by pixel-area
    area_from_props = array_to_process_props.area

    return area_as_poi_number, area_from_props


def get_areas_of_regions_in_mask(label_img, roi__mask=None, transform_to_label_img=False, binarization_threshold=0):
    """
    Returns a list of the areas of each separate region of input_array. The area is measured as pixels scaled by pixel-area. Refer to
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops for further documentation.

    Inputs:
    - label_img. ndarray. It can either be a label image (an image where pixels of separate regions are assigned the same value and a unique value is assigned to each separate region)
    or a binary image. If a label image is provided, the values must be >=0 and pixels of value 0 are assumed to correspond to the background. If a binary image is given,
    the parameter transform_to_label_img must be set to True in order for the function to transfom it in a label image using skimage.measure.label method
    (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label). If a binary mask is provided it will be firstly binarized and set in the 0-1 range using the
    binarization_threshold parameter. Pixels with intensity values >binarization_threshold will be set to 1 and considered pixels of interest, while the rest will be
    set to 0 and considered background. The default value for binarization_threshold is 0. NOTE: a binary mask can be considered a label image if only a single
    region is present.
    - roi__mask. Optional. ndarray of the same size of input_array. Binary mask. If provided, restricts the analysis to a region of interest. The region of interest is assumed to
    the positive pixels in roi_mas_k.
    - binarization_threshold. Int or float. Only applies when transform_to_label_img is True. Default 0. Defines the highpass threshold to distinguish pixels of interest from background
    in label_img. Pixels whose value is >binarization_threshold are considered pixels of interest. The rest of the pixels are considered background.

    Output: list. The area in number of pixels scaled by pixel-area. Refer to https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops of each separate
    region of label_img.
    
    """
    if not transform_to_label_img:
        assert np.min(label_img)==0, 'label_img must have background values set to 0 if a label image is provided'
        assert np.max(label_img)>0, 'label_img must have label region values >0 if a label image is provided'

    #Copy label image
    label_img_copy = label_img.copy()

    #Transform the image in a label image, if transform_to_label is True
    if transform_to_label_img:
        #Threshold label_img_copy using binarization_threshold. Set the values to 1 and 0, where 1s are assumed to be the pixels of interest
        label_img_copy_01 = np.where(label_img_copy>binarization_threshold, 1,0)
        label_img_i = label(label_img_copy_01)
    else:
        label_img_i = label_img_copy
    
    #If roi_mask is provided, copy it and use it to set to 0 pixels in label_img_i corresponded by background pixels in roi__mask
    if hasattr(roi__mask, "__len__"):
        roi__mask_copy = roi__mask.copy()
        img_to_pro_cess = np.where(roi__mask_copy>0, label_img_i, 0)
    else:
        img_to_pro_cess = label_img_i

    #Get regions properties
    img_to_pro_cess_regionprops = regionprops(img_to_pro_cess)

    #Initialize the output list
    output_list = []

    #Iterate through the measurements of each region
    for reg_mes in img_to_pro_cess_regionprops:

        #Get the area of the region
        region_area = reg_mes.area

        #Append the area to the output list
        output_list.append(region_area)

    return output_list


def get_covex_hull_from_mask(input_mask, roi_mask=None, threshold_4mask=0):
    """
    Given:
    - a binary mask defining a region of interst. It is assumed that the region of intest are pixels whose value is >threshold_4mask (default 0).
    
    It returns:
    - position 0, the minimum convex hull for the region of interest. Refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html for the output.
    - position 1, the coordinates of the pixels of interest in input_mask. Pixels of interest are assumed to be all pixels in input_mask whose value is >threshold_4mask (default 0).

    The indeces of the coordinates of the vertices of the convex hull within the coordinates list (output-1), are accessible from output-0 as output-0.vertices (if input_mask is 2D,
    their are in counterclockwise order).
    The area of the the convex hull (perimeter if input_mask is 2D) is accessible as output-0.areas.
    The volume of the convex hull (area if input_mask is 2D) is accessible as output-0.volume.

    If roi_mask is provided (binary array of the same shape of input_mask), the analysis is restricted to the region of interest. The region of interst is
    assumed to correspond to the positive pixels in roi_mask.
    """
    #Copy input_mask, threshold it using threshold_4mask. Set its values to 1 and 0, where 1 are pixels of interest, 0 the background.
    input_mask_01 = np.where(input_mask>threshold_4mask, 1,0)

    #If roi_mask is provided, copy it and use it to set to 0 pixels in input_mask_01 which are corresponded by background pixels in roi_mask
    if hasattr(roi_mask, "__len__"):
        #Copy roi_mask
        roi_mask_copy = roi_mask.copy()

        #Set to 0 pixels in binary input mask which are corresponded by background pixels in roi_mask
        segmented_input_mask = np.where(roi_mask_copy>0,input_mask_01,0)
    else:
        segmented_input_mask = input_mask_01
    
    #Get the coordinates of the pixels of interest in segmented_input_mask
    mask_pixels_coords = np.argwhere(segmented_input_mask>0)
    
    return ConvexHull(mask_pixels_coords), mask_pixels_coords

