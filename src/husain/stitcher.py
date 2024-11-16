import glob
import cv2
import numpy as np
import os
from tqdm import tqdm

def detectFeaturesAndMatch(image1, image2, maxNumOfFeatures=30):
    siftDetector = cv2.SIFT_create()
    keypoints1, descriptors1 = siftDetector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = siftDetector.detectAndCompute(image2, None)
    bruteForceMatcher = cv2.BFMatcher(cv2.NORM_L2)
    rawMatches = bruteForceMatcher.match(descriptors1, descriptors2)
    sortedMatches = sorted(rawMatches, key=lambda x: x.distance)
    featureCorrespondences = []
    for match in sortedMatches:
        featureCorrespondences.append((keypoints1[match.queryIdx].pt, keypoints2[match.trainIdx].pt))
    sourcePoints = np.float32([point_pair[0] for point_pair in featureCorrespondences[:maxNumOfFeatures]]).reshape(-1, 1, 2)
    destinationPoints = np.float32([point_pair[1] for point_pair in featureCorrespondences[:maxNumOfFeatures]]).reshape(-1, 1, 2)
    return np.array(featureCorrespondences[:maxNumOfFeatures]), sourcePoints, destinationPoints

def calculateHomography(point_correspondences):
    matrix_A = []
    for correspondence in point_correspondences:
        src_point, dst_point = correspondence
        src_x, src_y = src_point[0], src_point[1]
        dst_x, dst_y = dst_point[0], dst_point[1]
        matrix_A.append([src_x, src_y, 1, 0, 0, 0, -dst_x * src_x, -dst_x * src_y, -dst_x])
        matrix_A.append([0, 0, 0, src_x, src_y, 1, -dst_y * src_x, -dst_y * src_y, -dst_y])
    matrix_A = np.asarray(matrix_A)
    U, S, Vh = np.linalg.svd(matrix_A)
    homography_elements = Vh[-1, :] / Vh[-1, -1]
    homography_matrix = homography_elements.reshape(3, 3)
    return homography_matrix

def getBestHomographyRANSAC(correspondences, trials=1000, threshold=10, num_samples=4):
    best_homography = None
    max_inliers = []
    for i in tqdm(range(trials)):
        selected_indices = np.random.choice(len(correspondences), num_samples, replace=False)
        random_sample = correspondences[selected_indices]
        estimated_homography = calculateHomography(random_sample)
        current_inliers = []
        for correspondence in correspondences:
            src_point = np.append(correspondence[0], 1)
            dst_point = np.append(correspondence[1], 1)
            estimated_dst = np.dot(estimated_homography, src_point)
            estimated_dst /= estimated_dst[-1]
            error = np.linalg.norm(dst_point - estimated_dst)
            if error < threshold:
                current_inliers.append(correspondence)
        if len(current_inliers) > len(max_inliers):
            max_inliers = current_inliers
            best_homography = estimated_homography
    return best_homography, random_sample

def computeBoundingBoxOfWarpedImage(homography_matrix, img_width, img_height):
    original_corners = np.array([[0, img_width - 1, 0, img_width - 1], [0, 0, img_height - 1, img_height - 1], [1, 1, 1, 1]])
    transformed_corners = np.dot(homography_matrix, original_corners)
    transformed_corners /= transformed_corners[2, :]
    x_min = np.min(transformed_corners[0])
    x_max = np.max(transformed_corners[0])
    y_min = np.min(transformed_corners[1])
    y_max = np.max(transformed_corners[1])
    return int(x_min), int(x_max), int(y_min), int(y_max)

def warpAndPlaceSourceImage(source_img, homography_matrix, dest_img, use_forward_mapping=False, offset=(0, 0)):
    """Warps the source image onto the destination image using forward or inverse mapping."""

    height, width, _ = source_img.shape
    homography_inv = np.linalg.inv(homography_matrix)

    if use_forward_mapping:
        # Forward mapping
        coords = np.indices((width, height)).reshape(2, -1)
        homogeneous_coords = np.vstack((coords, np.ones(coords.shape[1])))
        transformed_coords = np.dot(homography_matrix, homogeneous_coords)
        transformed_coords /= transformed_coords[2, :]

        x_output, y_output = transformed_coords.astype(np.int32)[:2, :]

        valid_indices = (x_output >= 0) & (x_output < dest_img.shape[1]) & \
                        (y_output >= 0) & (y_output < dest_img.shape[0])

        x_output = x_output[valid_indices] + offset[0]
        y_output = y_output[valid_indices] + offset[1]
        x_input = coords[0][valid_indices]
        y_input = coords[1][valid_indices]

        dest_img[y_output, x_output] = source_img[y_input, x_input]

    else:  # Inverse mapping
        x_min, x_max, y_min, y_max = computeBoundingBoxOfWarpedImage(homography_matrix, width, height)

        x_coords, y_coords = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        coords = np.vstack((x_coords.ravel(), y_coords.ravel(), np.ones(x_coords.size)))

        transformed_coords = np.dot(homography_inv, coords)
        transformed_coords /= transformed_coords[2, :]

        x_input = transformed_coords[0].astype(np.int32)
        y_input = transformed_coords[1].astype(np.int32)

        valid = (x_input >= 0) & (x_input < width) & (y_input >= 0) & (y_input < height)
        final_x = x_coords.ravel() + offset[0]
        final_y = y_coords.ravel() + offset[1]
        valid &= (final_x >= 0) & (final_x < dest_img.shape[1]) & \
                 (final_y >= 0) & (final_y < dest_img.shape[0])

        valid_indices = np.where(valid)[0]
        if not valid_indices.size:
            print("No valid coordinates found after applying offset and boundary checks.")
            return

        dest_img[final_y[valid_indices], final_x[valid_indices]] = source_img[y_input[valid_indices], x_input[valid_indices]]


class ImageBlenderWithPyramids():
    def __init__(self, pyramid_depth=6):
        self.pyramid_depth = pyramid_depth

    def getGaussianPyramid(self, image):
        pyramid = [image]
        for _ in range(self.pyramid_depth - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid

    def getLaplacianPyramid(self, image):
        pyramid = []
        for _ in range(self.pyramid_depth - 1):
            next_level_image = cv2.pyrDown(image)
            upsampled_image = cv2.pyrUp(next_level_image, dstsize=(image.shape[1], image.shape[0]))
            pyramid.append(image.astype(float) - upsampled_image.astype(float))
            image = next_level_image
        pyramid.append(image)
        return pyramid

    def getBlendingPyramid(self, laplacian_a, laplacian_b, gaussian_mask_pyramid):
        blended_pyramid = []
        for i, mask in enumerate(gaussian_mask_pyramid):
            triplet_mask = cv2.merge((mask, mask, mask))
            blended_pyramid.append(laplacian_a[i] * triplet_mask + laplacian_b[i] * (1 - triplet_mask))
        return blended_pyramid

    def reconstructFromPyramid(self, laplacian_pyramid):
        reconstructed_image = laplacian_pyramid[-1]
        for laplacian_level in reversed(laplacian_pyramid[:-1]):
            reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=laplacian_level.shape[:2][::-1]).astype(float) + laplacian_level.astype(float)
        return reconstructed_image

    def generateMaskFromImage(self, image):
        mask = np.all(image != 0, axis=2)
        mask_image = np.zeros(image.shape[:2], dtype=float)
        mask_image[mask] = 1.0
        return mask_image

    def blendImages(self, image1, image2):
        laplacian1 = self.getLaplacianPyramid(image1)
        laplacian2 = self.getLaplacianPyramid(image2)
        mask1 = self.generateMaskFromImage(image1).astype(np.bool_)
        mask2 = self.generateMaskFromImage(image2).astype(np.bool_)
        overlap_region = mask1 & mask2
        y_coords, x_coords = np.where(overlap_region)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        final_mask = np.zeros(image1.shape[:2])
        final_mask[:, :(min_x + max_x) // 2] = 1.0
        gaussian_mask_pyramid = self.getGaussianPyramid(final_mask)
        blended_pyramid = self.getBlendingPyramid(laplacian1, laplacian2, gaussian_mask_pyramid)
        blended_image = self.reconstructFromPyramid(blended_pyramid)
        return blended_image, mask1, mask2


class PanaromaStitcher:
    """Stitches images together to create a panaroma."""

    def __init__(self,  shape=(600, 400), threshold = 2, trials = 3000, offset=(2300, 800)):
        self.shape = shape
        self.threshold = threshold
        self.trials = trials
        self.offset = offset
        self.all_images = None
        self.ImageSet_id = None


    def make_panaroma_for_images_in(self, path):
        """Creates a panaroma from images in a directory."""
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        
        self.all_images = all_images
        
        print("sort_images: ",all_images)
        print('Found {} Images for stitching'.format(len(all_images)))

        if len(all_images) < 2:
            raise ValueError("At least two images are required for stitching.")


        dir_name = os.path.basename(os.path.dirname(path))
        scene_id = ''.join(filter(str.isdigit, dir_name))
        self.ImageSet_id = int(scene_id) if scene_id.isdigit() else 1
        
        output_path = f'outputs/I{self.ImageSet_id}/custom'
        os.makedirs(output_path, exist_ok=True)
        
        prevH = np.eye(3)
        prevH = self._stitch_imgs(num_images = len(all_images))
        
        print("warping complete for set", self.ImageSet_id)

        # WARPING COMPLETE. BLENDING START
        print("We are starting with blending")
        b = ImageBlenderWithPyramids()
        finalImg = cv2.imread(f'outputs/I{self.ImageSet_id}/custom/warped_0.png')

        for index in range(1, len(all_images)):
            print('blending', index)
            
            img2 = cv2.imread(f'outputs/I{self.ImageSet_id}/custom/warped_{index}.png')
            finalImg, mask1truth, mask2truth = b.blendImages(finalImg, img2)
            mask1truth = mask1truth + mask2truth

        if not os.path.exists(f'outputs/I{self.ImageSet_id}/custom'):
            os.makedirs(f'outputs/I{self.ImageSet_id}/custom')
        cv2.imwrite(f'outputs/I{self.ImageSet_id}/custom/blended_image.png', finalImg)

        return finalImg, prevH
    
    def _stitch_imgs(self, num_images):

        """
        Generalized stitching function for stitching an arbitrary number of images.
        Args:
            num_images (int): Number of images to stitch.
        Returns:
            np.ndarray: Final homography matrix after stitching.
        """

        
        H = np.eye(3)   # Initialize the homography matrix

        # First pass: stitch left to the middle sequentially
        middle_idx = num_images // 2  # Middle index for reference
        for i in range(middle_idx - 1, -1, -1):  # Sequentially stitch left to middle
            H = self.stitch_and_save_images(i + 1, i, H)

        # Reset homography for the next pass
        H = np.eye(3)

        H = self.stitch_and_save_images(middle_idx, middle_idx, H)
        # Second pass: stitch right to the middle sequentially
        for i in range(middle_idx, num_images - 1):  # Sequentially stitch right to middle
            H = self.stitch_and_save_images(i, i + 1, H)

        return H
    
    def stitch_and_save_images(self, src_idx, dest_idx, prev_homography):
        warped_image = np.zeros((3000, 6000, 3), dtype=np.uint8)
        src_img = cv2.imread(self.all_images[src_idx])
        dest_img = cv2.imread(self.all_images[dest_idx])
        resized_src_img = cv2.resize(src_img, self.shape)
        resized_dest_img = cv2.resize(dest_img, self.shape)
        matches, src_pts, dest_pts = detectFeaturesAndMatch(resized_dest_img, resized_src_img)
        best_homography, _ = getBestHomographyRANSAC(matches, trials=self.trials, threshold=self.threshold)
        new_cumulative_homography = np.dot(prev_homography, best_homography)
        warpAndPlaceSourceImage(resized_dest_img, new_cumulative_homography, dest_img=warped_image, offset=self.offset)
        cv2.imwrite(f'outputs/I{self.ImageSet_id}/custom/warped_{dest_idx}.png', warped_image)
        return new_cumulative_homography
