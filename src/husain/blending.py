import numpy as np
import cv2
from typing import List, Tuple

class ImageBlender:
    """Blend images using Laplacian pyramid blending."""
    
    def __init__(self, pyramid_depth: int = 6):
        self.pyramid_depth = pyramid_depth
    
    def get_gaussian_pyramid(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate Gaussian pyramid for an image."""
        pyramid = [image]
        for _ in range(self.pyramid_depth - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid
    
    def get_laplacian_pyramid(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate Laplacian pyramid for an image."""
        pyramid = []
        for _ in range(self.pyramid_depth - 1):
            next_img = cv2.pyrDown(image)
            upsampled = cv2.pyrUp(next_img, dstsize=(image.shape[1], image.shape[0]))
            pyramid.append(image.astype(float) - upsampled.astype(float))
            image = next_img
        pyramid.append(image)
        return pyramid
    
    def blend_pyramids(self, 
                      laplacian_a: List[np.ndarray],
                      laplacian_b: List[np.ndarray],
                      mask_pyramid: List[np.ndarray]) -> List[np.ndarray]:
        """Blend two Laplacian pyramids using a Gaussian mask pyramid."""
        return [
            la * cv2.merge((m,m,m)) + lb * (1 - cv2.merge((m,m,m)))
            for la, lb, m in zip(laplacian_a, laplacian_b, mask_pyramid)
        ]
    
    def reconstruct_from_pyramid(self, pyramid: List[np.ndarray]) -> np.ndarray:
        """Reconstruct image from Laplacian pyramid."""
        image = pyramid[-1]
        for level in reversed(pyramid[:-1]):
            size = level.shape[:2][::-1]
            image = cv2.pyrUp(image, dstsize=size).astype(float) + level
        return image
    
    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate binary mask from image."""
        mask = np.all(image != 0, axis=2)
        result = np.zeros(image.shape[:2], dtype=float)
        result[mask] = 1.0
        return result
    
    def blend_images(self, image1: np.ndarray, 
                    image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Blend two images using pyramid blending."""
        lap1 = self.get_laplacian_pyramid(image1)
        lap2 = self.get_laplacian_pyramid(image2)
        
        mask1 = self.generate_mask(image1).astype(bool)
        mask2 = self.generate_mask(image2).astype(bool)
        
        overlap = mask1 & mask2
        y_coords, x_coords = np.where(overlap)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        
        final_mask = np.zeros(image1.shape[:2])
        final_mask[:, :(min_x + max_x) // 2] = 1.0
        
        mask_pyramid = self.get_gaussian_pyramid(final_mask)
        blended_pyramid = self.blend_pyramids(lap1, lap2, mask_pyramid)
        blended = self.reconstruct_from_pyramid(blended_pyramid)
        
        return blended, mask1, mask2