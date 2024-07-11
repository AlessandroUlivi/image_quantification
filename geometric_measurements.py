import numpy as np
from scipy.spatial import ConvexHull


def get_mask_area():
    return


def get_areas_of_regions_in_mask():
    return


def get_covex_hull_coordinates_from_mask(input_mask, roi_mask=None, threshold_4mask=0):
    """
    Given:
    - a binary mask defining a region of interst.
    
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

