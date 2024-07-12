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
    #Copy input_array and binarize it in the 0 and 1 range
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
    array_to_process_props = regionprops(array_to_process)

    #Get the area of the input_array region of interest number of pixels scaled by pixel-area
    area_from_props = array_to_process.area

    return area_as_poi_number, area_from_props


def get_areas_of_regions_in_mask():
    return


def get_covex_hull_coordinates_from_mask(input_mask, roi_mask=None, threshold_4mask=0):
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
    #Copy input_mask and rescale the values in the 1, 0 range where 1 are pixels of interest, 0 the background. Pixels whose value is >threshold_4mask are assumed to
    # be the pixels of interes
    rescaled_input_mask = np.where(input_mask>threshold_4mask, 1,0)

    #If roi_mask is provided, copy it and use it to set to 0 pixels in rescaled_input_mask which are corresponded by background pixels in roi_mask
    if hasattr(roi_mask, "__len__"):
        #Copy roi_mask
        roi_mask_copy = roi_mask.copy()

        #Set to 0 pixels in rescaled_input_mask which are corresponded by background pixels in roi_mask
        segmented_input_mask = np.where(roi_mask_copy>0,rescaled_input_mask,0)
    else:
        segmented_input_mask = rescaled_input_mask
    
    #Get the coordinates of the pixels of interest in segmented_input_mask
    mask_pixels_coords = np.argwhere(segmented_input_mask>0)
    
    return ConvexHull(mask_pixels_coords), mask_pixels_coords

