import numpy as np
import cv2
from typing import Tuple

def compute_warped_bounds(homography: np.ndarray, 
                         width: int, 
                         height: int) -> Tuple[int, int, int, int]:
    """
    Compute the bounding box of the warped image.
    
    Args:
        homography: Homography matrix
        width: Image width
        height: Image height
        
    Returns:
        Tuple of (x_min, x_max, y_min, y_max)
    """
    corners = np.array([
        [0, width-1, 0, width-1],
        [0, 0, height-1, height-1],
        [1, 1, 1, 1]
    ])
    
    transformed = np.dot(homography, corners)
    transformed /= transformed[2, :]
    
    return (int(np.min(transformed[0])), int(np.max(transformed[0])),
            int(np.min(transformed[1])), int(np.max(transformed[1])))

def warp_and_place_image(source_img: np.ndarray,
                        homography: np.ndarray,
                        dest_img: np.ndarray,
                        use_forward_mapping: bool = False,
                        offset: Tuple[int, int] = (0, 0)) -> None:
    """
    Warp source image and place it on destination image using specified mapping.
    
    Args:
        source_img: Source image to warp
        homography: Homography matrix
        dest_img: Destination image to place warped image on
        use_forward_mapping: Whether to use forward mapping
        offset: Offset for placement (x, y)
    """
    height, width = source_img.shape[:2]
    H_inv = np.linalg.inv(homography)

    if use_forward_mapping:
        coords = np.indices((width, height)).reshape(2, -1)
        homog_coords = np.vstack((coords, np.ones(coords.shape[1])))
        
        transformed = np.dot(homography, homog_coords)
        transformed /= transformed[2, :]
        
        x_out, y_out = transformed.astype(np.int32)[:2, :]
        valid = ((x_out >= 0) & (x_out < dest_img.shape[1]) &
                (y_out >= 0) & (y_out < dest_img.shape[0]))
        
        x_out = x_out[valid] + offset[0]
        y_out = y_out[valid] + offset[1]
        x_in = coords[0][valid]
        y_in = coords[1][valid]
        
        dest_img[y_out, x_out] = source_img[y_in, x_in]
    else:
        x_min, x_max, y_min, y_max = compute_warped_bounds(homography, width, height)
        x_coords, y_coords = np.meshgrid(
            np.arange(x_min, x_max),
            np.arange(y_min, y_max)
        )
        
        coords = np.vstack((x_coords.ravel(), y_coords.ravel(), np.ones(x_coords.size)))
        transformed = np.dot(H_inv, coords)
        transformed /= transformed[2, :]
        
        x_in = transformed[0].astype(np.int32)
        y_in = transformed[1].astype(np.int32)
        
        valid = ((x_in >= 0) & (x_in < width) & 
                (y_in >= 0) & (y_in < height))
        
        final_x = x_coords.ravel() + offset[0]
        final_y = y_coords.ravel() + offset[1]
        
        valid &= ((final_x >= 0) & (final_x < dest_img.shape[1]) &
                 (final_y >= 0) & (final_y < dest_img.shape[0]))
        
        valid_idx = np.where(valid)[0]
        if valid_idx.size:
            dest_img[final_y[valid_idx], final_x[valid_idx]] = \
                source_img[y_in[valid_idx], x_in[valid_idx]]