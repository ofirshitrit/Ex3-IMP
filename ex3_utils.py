import sys
from typing import List
from scipy.ndimage import convolve
import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 324249150

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    image_shape = im1.shape
    if len(image_shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        plt.gray()
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        plt.gray()
    if win_size % 2 == 0:
        return "win_size must be an odd number"
    half_win_size = win_size // 2

    kernel_x = np.array([[-1, 0, 1]])
    kernel_y = kernel_x.T

    Ix = cv2.filter2D(im2, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)
    It = im2 - im1

    original_points = []  # list for the original points.
    vec_per_point = []  # list for the OP vectors for each point in @original_points.

    # Here I will go over a blocks in the shape of (step_size x step_size)
    # and find the optical flow vector for each block.
    # optional start from @half_win_size + 1
    for row in range(step_size, image_shape[0] - half_win_size + 1, step_size):
        for col in range(step_size, image_shape[1] - half_win_size + 1, step_size):
            Ix_windowed = Ix[
                          row - half_win_size: row + half_win_size + 1,
                          col - half_win_size: col + half_win_size + 1,
                          ].flatten()
            Iy_windowed = Iy[
                          row - half_win_size: row + half_win_size + 1,
                          col - half_win_size: col + half_win_size + 1,
                          ].flatten()
            It_windowed = It[
                          row - half_win_size: row + half_win_size + 1,
                          col - half_win_size: col + half_win_size + 1,
                          ].flatten()
            A = np.vstack((Ix_windowed, Iy_windowed)).T  # A = [Ix, Iy]
            b = (A.T @ (-1 * It_windowed).T).reshape(2, 1)
            ATA = A.T @ A

            ATA_eig_vals = np.sort(np.linalg.eigvals(ATA))
            if ATA_eig_vals[0] <= 1 or ATA_eig_vals[1] / ATA_eig_vals[0] >= 100:
                # vec_per_point.append(np.array([0, 0]))
                # original_points.append([col, row])
                continue

            ATA_INV = np.linalg.inv(ATA)
            curr_vec = ATA_INV @ b
            original_points.append([col, row])
            vec_per_point.append([curr_vec[0, 0], curr_vec[1, 0]])

    return np.array(original_points),np.array(vec_per_point)
    pass


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    # Create image pyramids
    pyramid1 = createPyramid(img1, k)
    pyramid2 = createPyramid(img2, k)

    # Initialize the flow variables
    Uk = np.zeros_like(pyramid1[0], dtype=np.float32)
    Vk = np.zeros_like(pyramid1[0], dtype=np.float32)

    # Iterate over the pyramid levels in reverse order
    for i in range(k - 1, -1, -1):
        # Upsample the current flow estimates
        Uk = cv2.resize(Uk, (pyramid1[i].shape[1], pyramid1[i].shape[0])) * 2.0
        Vk = cv2.resize(Vk, (pyramid1[i].shape[1], pyramid1[i].shape[0])) * 2.0

        # Compute the optical flow for the current pyramid level
        flow = opticalFlowIterative(pyramid1[i], pyramid2[i], Uk, Vk, stepSize, winSize)

        # Update the flow estimates
        Uk += flow[..., 0]
        Vk += flow[..., 1]

    return np.stack((Uk, Vk), axis=-1)
    pass
def opticalFlowIterative(image1, image2, Uk, Vk, stepSize, winSize):
    # Calculate the spatial gradients of the second image
    Ix = cv2.Sobel(image2, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image2, cv2.CV_64F, 0, 1, ksize=3)

    # Iterate over each pixel and estimate optical flow
    flow = np.zeros((image1.shape[0], image1.shape[1], 2), dtype=np.float32)
    for i in range(0, image1.shape[0], stepSize):
        for j in range(0, image1.shape[1], stepSize):
            # Calculate the gradient of the template window
            window = image1[max(i - winSize // 2, 0):min(i + winSize // 2 + 1, image1.shape[0]),
                      max(j - winSize // 2, 0):min(j + winSize // 2 + 1, image1.shape[1])]
            Ix_window = Ix[max(i - winSize // 2, 0):min(i + winSize // 2 + 1, image1.shape[0]),
                         max(j - winSize // 2, 0):min(j + winSize // 2 + 1, image1.shape[1])]
            Iy_window = Iy[max(i - winSize // 2, 0):min(i + winSize // 2 + 1, image1.shape[0]),
                         max(j - winSize // 2, 0):min(j + winSize // 2 + 1, image1.shape[1])]

            # Calculate the error image
            error = window - image2[i, j]

            # Calculate the optical flow update
            A = np.column_stack((Ix_window.flatten(), Iy_window.flatten()))
            b = error.flatten()
            flow_update = np.linalg.lstsq(A, b, rcond=None)[0]

            # Update the optical flow estimates
            flow[i, j] = Uk[i, j] + flow_update[0]
            flow[i, j] = Vk[i, j] + flow_update[1]

    return flow

def createPyramid(img, k):
    pyramid = [img]
    for i in range(1, k):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid




# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pass


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pass


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pass


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    pass

