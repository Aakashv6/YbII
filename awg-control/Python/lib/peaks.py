import numpy as np
from typing import List
import scipy.ndimage as ndimage
import cv2

class LabelModel:
    label_name: str
    optional_params: set[str]
    def __init__(self) -> None:
        pass
    
    def find_peaks(self, img: np.ndarray) -> tuple[list, np.ndarray]:
        raise NotImplementedError("find_peaks method is not implemented")
    
class ScipyFindObj(LabelModel):
    label_name = "scipy.ndimage.find_objects"
    optional_params = {""}
    def __init__(
        self,
        peakfilter_size: tuple[int, int],
        threshold: float,
    ) -> None:
        self.peakfilter_size = peakfilter_size
        self.threshold = threshold
        
    def find_peaks(
        self, 
        img: np.ndarray,
    ) -> tuple[list, np.ndarray]:
        proc = img.copy()
        peakfilter_size = self.peakfilter_size
        threshold = self.threshold
        img_max = ndimage.maximum_filter(proc, size=peakfilter_size)
        img_min = ndimage.minimum_filter(proc, size=peakfilter_size)
        diff = (img_max - img_min) > threshold
        img_max[diff == 0] = 0
        labeled, num_objs = ndimage.label(img_max)
        img_slices = ndimage.find_objects(labeled)
        xc = []
        yc = []
        print(f"{num_objs} peaks found")
        for k, slice in enumerate(img_slices):
            xc.append(int(slice[0].start + (slice[0].stop - slice[0].start) / 2))
            yc.append(int(slice[1].start + (slice[1].stop - slice[1].start) / 2))
            cv2.rectangle(
                proc,
                (slice[1].start, slice[0].start),
                (slice[1].stop, slice[0].stop),
                color=(255, 0, 0),
                thickness=5,
            )
        peak_locs = np.sort(np.array([xc,yc]).T, axis=0)
        return peak_locs, proc

class Cv2FeatureFinder(LabelModel):
    label_name = "cv2.connectedComponentsWithStats"
    optional_params = {
        "gaussian_blur_params: dict{ksize: tuple[int, int], sigmaX: float, sigmaY: float}",
        "threshold_params: dict{adaptiveMethod: int, thresholdType: int, blockSize: int, C: int}",
        "connectivity: int",
    }
    def __init__(
        self,
        filter_area: float,
        filter_width: float,
        filter_height: float,
        gaussian_blur_params: dict=None,
        threshold_params: dict=None,
        connectivity: int=8,
    ) -> None:
        if gaussian_blur_params is None:
            self.gaussian_blur_params = {
                "ksize": (3,3),
                "sigmaX": 0.0,
                "sigmaY": 0.0,
            }
        else:
            self.gaussian_blur_params = gaussian_blur_params
        if threshold_params is None:
            self.threshold_params = {
                "adaptiveMethod": cv2.ADAPTIVE_THRESH_MEAN_C,
                "thresholdType": cv2.THRESH_BINARY,
                "blockSize": 3,
                "C": 0,
            }
        else:
            self.threshold_params = threshold_params
        self.connectivity = connectivity
        self.filter_area = filter_area
        self.filter_width = filter_width
        self.filter_height = filter_height
        
    def find_peaks(
        self, 
        img: np.ndarray,
    ) -> tuple[list, np.ndarray]:
        
        proc = img.copy()
        proc = proc.astype(np.uint8)
        proc = cv2.GaussianBlur(
            proc,
            self.gaussian_blur_params["ksize"],
            self.gaussian_blur_params["sigmaX"],
            self.gaussian_blur_params["sigmaY"],
        )
        proc = cv2.adaptiveThreshold(
            proc,
            255,
            adaptiveMethod=self.threshold_params["adaptiveMethod"],
            thresholdType=self.threshold_params["thresholdType"],
            blockSize=self.threshold_params["blockSize"],
            C=self.threshold_params["C"],
        )
        num_feature, labels, stats, centroids = cv2.connectedComponentsWithStats(
            proc,
            connectivity=self.connectivity,
        )
        num_feature -= 1
        # labels = labels[1:]
        stats = stats[1:]
        centroids = centroids[1:]
        peaks = []
        for k, (stat,center) in enumerate(zip(stats, centroids)):
            left, top, width, height, area = stat
            x = int(center[0])
            y = int(center[1])
            if area >= self.filter_area \
                and width >= self.filter_width \
                and height >= self.filter_height:
                cv2.rectangle(
                    proc,
                    (left, top),
                    (left + width, top + height),
                    color=(255, 0, 0),
                    thickness=5,
                )
                peaks.append([x,y])
        # peaks.sort(key=lambda xy: xy[0])
        return peaks, proc
    
class Cv2HoughCircles(LabelModel):
    label_name = "cv2.HoughCircles"
    optional_params = {
        "gaussian_blur_params: dict{ksize: tuple[int, int], sigmaX: float, sigmaY: float}",
    }
    def __init__(
        self,
        minDist: float,
        param1: float,
        param2: float,
        minRadius: int,
        maxRadius: int,
        dp: float=1.5,
        method: int=cv2.HOUGH_GRADIENT,
        gaussian_blur_params: dict=None,
    ) -> None:
        """
        Initialize hough circle detection on an image.

        Args:
            dp (float): Inverse ratio of the accumulator resolution, 1=the same as the image, 1.5 is default
            minDist (float): minimum distance between the centers of the detected circles
            param1 (float): (for HOUGH_GRADIENT) higher threshold for the Canny edge detector
            param2 (float): (for HOUGH_GRADIENT) threshold for circle detection, smaller for more circles
            minRadius (int): minimum circle radius
            maxRadius (int): maximum circle radius
            method (int, optional): Hough circle detection alg. Defaults to cv2.HOUGH_GRADIENT.
        """
        self.dp = dp
        self.minDist = minDist
        self.param1 = param1
        self.param2 = param2
        self.minRadius = minRadius
        self.maxRadius = maxRadius
        self.method = method
        if gaussian_blur_params is None:
            self.gaussian_blur_params = {
                "ksize": (3,3),
                "sigmaX": 0.0,
                "sigmaY": 0.0,
            }
        else:
            self.gaussian_blur_params = gaussian_blur_params
    
    def find_peaks(
        self, 
        img: np.ndarray,
    ) -> tuple[List, np.ndarray]:
        proc = img.copy()
        proc = proc.astype(np.uint8)
        proc = cv2.GaussianBlur(
            proc,
            self.gaussian_blur_params["ksize"],
            self.gaussian_blur_params["sigmaX"],
            self.gaussian_blur_params["sigmaY"],
        )
        circles = cv2.HoughCircles(
            proc,
            method=self.method,
            dp=self.dp,
            minDist=self.minDist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.minRadius,
            maxRadius=self.maxRadius,
        )
        circles = np.uint16(np.around(circles))
        peaks = []
        for c in circles[0, :]:
            cv2.circle(proc, (c[0], c[1]), c[2], (255, 0, 0), 3)
            peaks.append([c[0],c[1]])
        return peaks, proc