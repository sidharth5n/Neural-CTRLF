import numpy as np
import cv2
from misc.utils import unique_boxes

def generate_region_proposals(f):
    """
    Parameters
    ----------
    f : list
        Absolute path to the images

    Returns
    -------
    region_proposals : numpy.ndarray of shape (P, 4)
                       Region proposals in (x, y, w, h) format.
    """
    img = cv2.imread(f, 0)        
    m = img.mean()
    threshold_range = np.arange(0.6, 1.01, 0.1) * m
    C_range = range(3, 50, 2) #horizontal range
    R_range = range(3, 50, 2) #vertical range
    region_proposals = find_regions(img, threshold_range, C_range, R_range) #[img1, img2, ...]
    region_proposals, _ = unique_boxes(region_proposals)
    return region_proposals

def find_regions(img, threshold_range, C_range, R_range):
    """
    Finds region proposals in an image using different thresholding techniques 
    and different morphology kernel sizes.

    Parameters
    ----------
    img             : numpy.ndarray of shape (H, W)
                      Grayscale image
    threshold_range : numpy.ndarray of shape (M, )
                      Threshold values
    C_range         : numpy.ndarray of shape (L, )
                      Horizontal range
    R_range         : numpy.ndarray of shape (K, )
                      Vertical range
    
    Returns
    -------
    regions         : list
                      Region proposals from all the thresholded images.
    """

    ims = [cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1] for thresh in threshold_range]
    regions = []
    for t_img in ims:
        regions += extract_regions(t_img, C_range, R_range)
    return regions

def extract_regions(t_img, C_range, R_range):
    """
    Extracts region propsals for a given image using morphological transformation
    and connected component analysis.

    Parameters
    ---------
    t_img     : numpy.ndarray of shape (H, W)
                Thresholded binary image
    C_range   : numpy.ndarray of shape (L, )
                Width of morphology kernel
    R_range   : numpy.ndarray of shape (K, )
                Height of morphology kernel
    
    Returns
    -------
    all_boxes : list
                Region proposals using various shapes of the kernel.
                Each element is a list having the bbox info in (x, y, w, h) format.
    """
    all_boxes = []    
    for R in R_range:
        for C in C_range:
            s_img = cv2.morphologyEx(t_img, cv2.MORPH_CLOSE, np.ones((R, C), dtype = np.ubyte))
            n, l_img, stats, centroids = cv2.connectedComponentsWithStats(s_img, connectivity=4)
            boxes = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in stats]
            all_boxes += boxes
                
    return all_boxes