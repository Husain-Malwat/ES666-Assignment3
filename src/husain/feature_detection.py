import cv2
import numpy as np
from typing import Tuple, List

def detect_features_and_match(image1: np.ndarray, 
                            image2: np.ndarray, 
                            max_features: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect and match features between two images using SIFT.
    
    Args:
        image1: First input image
        image2: Second input image
        max_features: Maximum number of features to match
        
    Returns:
        Tuple containing feature correspondences, source points, and destination points
    """
    sift_detector = cv2.SIFT_create()
    keypoints1, descriptors1 = sift_detector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift_detector.detectAndCompute(image2, None)
    
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    correspondences = [(keypoints1[m.queryIdx].pt, keypoints2[m.trainIdx].pt) 
                      for m in matches[:max_features]]
    
    src_points = np.float32([p[0] for p in correspondences]).reshape(-1, 1, 2)
    dst_points = np.float32([p[1] for p in correspondences]).reshape(-1, 1, 2)
    
    return np.array(correspondences), src_points, dst_points