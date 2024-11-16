import numpy as np
from typing import List, Tuple
from tqdm import tqdm

def calculate_homography(point_correspondences: np.ndarray) -> np.ndarray:
    """
    Calculate the homography matrix from point correspondences.
    
    Args:
        point_correspondences: Array of point pairs between images
        
    Returns:
        3x3 homography matrix
    """
    matrix_A = []
    for src_point, dst_point in point_correspondences:
        sx, sy = src_point
        dx, dy = dst_point
        matrix_A.extend([
            [sx, sy, 1, 0, 0, 0, -dx*sx, -dx*sy, -dx],
            [0, 0, 0, sx, sy, 1, -dy*sx, -dy*sy, -dy]
        ])
    
    matrix_A = np.asarray(matrix_A)
    _, _, Vh = np.linalg.svd(matrix_A)
    H = Vh[-1, :] / Vh[-1, -1]
    return H.reshape(3, 3)

def get_best_homography_ransac(correspondences: np.ndarray,
                             trials: int = 1000,
                             threshold: float = 10,
                             num_samples: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the best homography matrix using RANSAC.
    
    Args:
        correspondences: Array of point correspondences
        trials: Number of RANSAC iterations
        threshold: Error threshold for inlier detection
        num_samples: Number of samples for each iteration
        
    Returns:
        Tuple of best homography matrix and random sample used
    """
    best_homography = None
    max_inliers = []
    
    for _ in tqdm(range(trials), desc="RANSAC Progress"):
        indices = np.random.choice(len(correspondences), num_samples, replace=False)
        sample = correspondences[indices]
        H = calculate_homography(sample)
        
        inliers = []
        for correspondence in correspondences:
            src = np.append(correspondence[0], 1)
            dst = np.append(correspondence[1], 1)
            
            est_dst = np.dot(H, src)
            est_dst /= est_dst[-1]
            
            if np.linalg.norm(dst - est_dst) < threshold:
                inliers.append(correspondence)
                
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_homography = H
            
    return best_homography, sample