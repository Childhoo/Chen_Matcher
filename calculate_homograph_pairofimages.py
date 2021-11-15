# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:07:52 2019

@author: chenlin
"""
import cv2
import numpy as np

FLANN_INDEX_LINEAR = 0
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_KMEANS = 2
FLANN_INDEX_COMPOSITE = 3
FLANN_INDEX_KDTREE_SINGLE = 4
FLANN_INDEX_HIERARCHICAL = 5
FLANN_INDEX_LSH = 6
FLANN_INDEX_SAVED = 254
FLANN_INDEX_AUTOTUNED = 255

def homography(img1, img2, visualize=False):
    """
    Finds Homography matrix from Image1 to Image2.
        Two images should be a plane and can change in viewpoint

    :param img1: Source image
    :param img2: Target image
    :param visualize: Flag to visualize the matched pixels and Homography warping
    :return: Homography matrix. (or) Homography matrix, Visualization image - if visualize is True
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=INDEX_PARAM_TREES)
    # number of times the trees in the index should be recursively traversed
    # Higher values gives better precision, but also takes more time
    sch_params = dict(checks=SCH_PARAM_CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, sch_params)

    matches = flann.knnMatch(desc1, desc2, k=2)
#    logging.debug('{} matches found'.format(len(matches)))

    # select good matches
    matches_arr = []
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
        matches_arr.append(m)

    if len(good_matches) < 4:
        raise (Exception('Not enough matches found'))
    else:
        logging.debug('{} of {} are good matches'.format(len(good_matches), len(matches)))

    src_pts = [kp1[m.queryIdx].pt for m in good_matches]
    src_pts = np.array(src_pts, dtype=np.float32).reshape((-1, 1, 2))
    dst_pts = [kp2[m.trainIdx].pt for m in good_matches]
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape((-1, 1, 2))

    homo, mask = cv2.findHomography(src_pts, dst_pts, cv.RANSAC, 5)

    if visualize:
        res = visualize_homo(img1, img2, kp1, kp2, matches, homo, mask)
        return homo, res

    return homo 
