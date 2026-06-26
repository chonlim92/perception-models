"""Rule-based and statistical weather classification from sensor data.

Classifies weather conditions (clear, rain, fog, snow, overcast) by analyzing
camera images, lidar point clouds, and radar returns. Individual sensor
classifications are fused using configurable weighted averaging.
"""

from typing import Dict, Optional

import numpy as np
from scipy.ndimage import convolve, uniform_filter


def _gabor_kernel(
    frequency: float,
    theta: float,
    sigma_x: float = 3.0,
    sigma_y: float = 3.0,
    n_stds: float = 3.0,
) -> np.ndarray:
    """Generate a real-valued Gabor filter kernel.

    Parameters
    ----------
    frequency : float
        Spatial frequency of the sinusoidal carrier (cycles per pixel).
    theta : float
        Orientation of the filter in radians.
    sigma_x : float
        Standard deviation along the x-axis of the Gaussian envelope.
    sigma_y : float
        Standard deviation along the y-axis of the Gaussian envelope.
    n_stds : float
        Number of standard deviations for the kernel extent.

    Returns
    -------
    np.ndarray
        2D Gabor filter kernel (real part).
    """
    x_max = int(np.ceil(max(np.abs(n_stds * sigma_x * np.cos(theta)),
                             np.abs(n_stds * sigma_y * np.sin(theta)), 1)))
    y_max = int(np.ceil(max(np.abs(n_stds * sigma_x * np.sin(theta)),
                             np.abs(n_stds * sigma_y * np.cos(theta)), 1)))

    y, x = np.mgrid[-y_max:y_max + 1, -x_max:x_max + 1]

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Gaussian envelope * cosine carrier
    gaussian = np.exp(
        -0.5 * ((x_theta ** 2) / (sigma_x ** 2) + (y_theta ** 2) / (sigma_y ** 2))
    )
    kernel = gaussian * np.cos(2.0 * np.pi * frequency * x_theta)
    return kernel


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize a dictionary of scores so that values sum to 1.0.

    If all scores are zero, distributes probability uniformly.
    """
    total = sum(scores.values())
    if total <= 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    return {k: v / total for k, v in scores.items()}


class WeatherClassifier:
    """Rule-based and statistical weather classifier using multi-sensor fusion.

    Analyzes camera images, lidar point clouds, and radar data to produce a
    probability distribution over weather conditions: clear, rain, fog, snow,
    and overcast.

    Parameters
    ----------
    camera_weight : float
        Fusion weight for camera-based classification (default 0.4).
    lidar_weight : float
        Fusion weight for lidar-based classification (default 0.35).
    radar_weight : float
        Fusion weight for radar-based classification (default 0.25).
    rain_gabor_threshold : float
        Gabor filter response threshold above which rain streaks are detected.
    fog_contrast_threshold : float
        Grayscale standard deviation below which fog is inferred.
    low_brightness_threshold : float
        Mean brightness below which night/overcast is inferred.
    high_brightness_threshold : float
        Mean brightness above which snow-covered scenes are considered.
    fog_range_ratio_threshold : float
        Ratio of far-range to near-range lidar density below which fog is inferred.
    radar_rcs_threshold : float
        RCS value (dBsm) below which a radar target is considered clutter.
    """

    WEATHER_CLASSES = ("clear", "rain", "fog", "snow", "overcast")

    def __init__(
        self,
        camera_weight: float = 0.4,
        lidar_weight: float = 0.35,
        radar_weight: float = 0.25,
        rain_gabor_threshold: float = 0.15,
        fog_contrast_threshold: float = 30.0,
        low_brightness_threshold: float = 80.0,
        high_brightness_threshold: float = 200.0,
        fog_range_ratio_threshold: float = 0.2,
        radar_rcs_threshold: float = -10.0,
    ) -> None:
        self.camera_weight = camera_weight
        self.lidar_weight = lidar_weight
        self.radar_weight = radar_weight
        self.rain_gabor_threshold = rain_gabor_threshold
        self.fog_contrast_threshold = fog_contrast_threshold
        self.low_brightness_threshold = low_brightness_threshold
        self.high_brightness_threshold = high_brightness_threshold
        self.fog_range_ratio_threshold = fog_range_ratio_threshold
        self.radar_rcs_threshold = radar_rcs_threshold

    # ------------------------------------------------------------------
    # Camera-based classification
    # ------------------------------------------------------------------

    def classify_from_camera(self, image: np.ndarray) -> Dict[str, float]:
        """Classify weather from a camera image.

        Converts to grayscale, then evaluates brightness, contrast, and Gabor
        filter response for rain streak detection.

        Parameters
        ----------
        image : np.ndarray
            Input image as HxW (grayscale) or HxWx3 (RGB/BGR) with dtype
            uint8 or float in [0, 1].

        Returns
        -------
        Dict[str, float]
            Normalized probability distribution over weather classes.
        """
        # Convert to float grayscale in [0, 255] range
        gray = self._to_grayscale(image)

        # Brightness analysis
        mean_brightness = float(np.mean(gray))

        # Contrast analysis (std of grayscale pixel values)
        contrast = float(np.std(gray))

        # Rain streak detection via Gabor filters at near-vertical orientations
        rain_response = self._detect_rain_streaks(gray)

        # Compute raw scores
        scores: Dict[str, float] = {k: 0.0 for k in self.WEATHER_CLASSES}

        # --- Clear ---
        # High contrast and moderate brightness suggests clear weather
        if contrast > self.fog_contrast_threshold:
            clear_score = contrast / 128.0  # normalized by half max std
            if self.low_brightness_threshold <= mean_brightness <= self.high_brightness_threshold:
                clear_score *= 1.5
            scores["clear"] = max(clear_score, 0.0)

        # --- Fog ---
        # Low contrast is the primary indicator of fog
        if contrast < self.fog_contrast_threshold:
            fog_score = 1.0 - (contrast / self.fog_contrast_threshold)
            scores["fog"] = fog_score * 2.0  # amplify signal

        # --- Rain ---
        # High Gabor filter response at near-vertical orientations
        if rain_response > self.rain_gabor_threshold:
            rain_score = rain_response / 0.5  # normalize to useful range
            scores["rain"] = min(rain_score, 2.0)

        # --- Snow ---
        # Very high mean brightness (snow-covered ground) and moderate contrast
        if mean_brightness > self.high_brightness_threshold:
            snow_score = (mean_brightness - self.high_brightness_threshold) / 55.0
            scores["snow"] = min(snow_score * 1.5, 2.0)

        # --- Overcast ---
        # Low brightness with moderate contrast (not fog)
        if mean_brightness < self.low_brightness_threshold and contrast >= self.fog_contrast_threshold * 0.5:
            overcast_score = (self.low_brightness_threshold - mean_brightness) / self.low_brightness_threshold
            scores["overcast"] = overcast_score * 1.5
        elif self.low_brightness_threshold <= mean_brightness <= 140.0 and contrast < 50.0:
            # Moderately dim with somewhat reduced contrast
            scores["overcast"] = 0.5

        return _normalize_scores(scores)

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert an image to float32 grayscale in [0, 255]."""
        if image.ndim == 2:
            gray = image.astype(np.float32)
        elif image.ndim == 3 and image.shape[2] == 3:
            # Standard luminance weights (ITU-R BT.601)
            gray = (
                0.2989 * image[:, :, 0].astype(np.float32)
                + 0.5870 * image[:, :, 1].astype(np.float32)
                + 0.1140 * image[:, :, 2].astype(np.float32)
            )
        elif image.ndim == 3 and image.shape[2] == 4:
            # RGBA - ignore alpha
            gray = (
                0.2989 * image[:, :, 0].astype(np.float32)
                + 0.5870 * image[:, :, 1].astype(np.float32)
                + 0.1140 * image[:, :, 2].astype(np.float32)
            )
        else:
            # Fallback: take the first channel
            gray = image[:, :, 0].astype(np.float32) if image.ndim == 3 else image.astype(np.float32)

        # Normalize to [0, 255] if input is float in [0, 1]
        if gray.max() <= 1.0 and gray.min() >= 0.0 and image.dtype in (np.float32, np.float64):
            gray *= 255.0

        return gray

    def _detect_rain_streaks(self, gray: np.ndarray) -> float:
        """Detect rain streaks using Gabor filters at near-vertical orientations.

        Applies Gabor filters at orientations between 70 and 110 degrees and
        returns the mean of the maximum filter responses across the image.

        Parameters
        ----------
        gray : np.ndarray
            Grayscale image as float32 in [0, 255].

        Returns
        -------
        float
            Normalized rain streak response (higher = more likely rain).
        """
        # Generate Gabor kernels at near-vertical orientations (70-110 degrees)
        orientations_deg = [70.0, 80.0, 85.0, 90.0, 95.0, 100.0, 110.0]
        frequency = 0.1  # spatial frequency suitable for rain streak width

        max_response = np.zeros_like(gray)

        for angle_deg in orientations_deg:
            theta = np.deg2rad(angle_deg)
            kernel = _gabor_kernel(
                frequency=frequency,
                theta=theta,
                sigma_x=2.0,
                sigma_y=4.0,
                n_stds=2.5,
            )
            response = convolve(gray, kernel, mode="reflect")
            max_response = np.maximum(max_response, np.abs(response))

        # Normalize by image intensity range to get a relative measure
        intensity_range = gray.max() - gray.min()
        if intensity_range < 1.0:
            return 0.0

        normalized = max_response / intensity_range

        # Use the 95th percentile of the response as the overall score
        # (avoids being dominated by a few outlier pixels)
        score = float(np.percentile(normalized, 95))
        return score

    # ------------------------------------------------------------------
    # Lidar-based classification
    # ------------------------------------------------------------------

    def classify_from_lidar(self, points: np.ndarray) -> Dict[str, float]:
        """Classify weather from a lidar point cloud.

        Analyzes point density at various ranges and intensity variance to
        infer weather conditions.

        Parameters
        ----------
        points : np.ndarray
            Nx3 or Nx4 array where columns are [x, y, z] or [x, y, z, intensity].

        Returns
        -------
        Dict[str, float]
            Normalized probability distribution over weather classes.
        """
        if points.ndim != 2 or points.shape[0] == 0:
            # Return uniform distribution for invalid input
            return {k: 1.0 / len(self.WEATHER_CLASSES) for k in self.WEATHER_CLASSES}

        has_intensity = points.shape[1] >= 4

        # Compute range for each point
        xyz = points[:, :3]
        ranges = np.sqrt(np.sum(xyz ** 2, axis=1))

        # Bin points by range
        near_mask = ranges < 20.0          # 0-20m
        mid_mask = (ranges >= 20.0) & (ranges < 50.0)   # 20-50m
        far_mask = (ranges >= 50.0) & (ranges < 100.0)  # 50-100m
        very_far_mask = ranges >= 100.0    # >100m

        near_count = float(np.sum(near_mask))
        mid_count = float(np.sum(mid_mask))
        far_count = float(np.sum(far_mask))
        very_far_count = float(np.sum(very_far_mask))

        total_points = float(points.shape[0])

        # Compute densities normalized by range-band volume (approximate)
        # Volume of a spherical shell: proportional to r^2 * dr
        # Near: 0-20m -> volume ~ 20^3/3 = ~2667
        # Mid: 20-50m -> volume ~ (50^3 - 20^3)/3 = ~38667
        # Far: 50-100m -> volume ~ (100^3 - 50^3)/3 = ~291667
        near_density = near_count / 2667.0 if near_count > 0 else 0.0
        mid_density = mid_count / 38667.0 if mid_count > 0 else 0.0
        far_density = far_count / 291667.0 if far_count > 0 else 0.0

        # Fog score: density drops sharply beyond 50m
        # Compare far-range density to near-range density
        if near_density > 0:
            density_ratio = far_density / near_density
        else:
            density_ratio = 1.0  # cannot determine

        fog_score = 0.0
        if density_ratio < self.fog_range_ratio_threshold:
            fog_score = 1.0 - (density_ratio / self.fog_range_ratio_threshold)

        # Additional fog signal: very few points beyond 50m relative to total
        far_fraction = (far_count + very_far_count) / total_points if total_points > 0 else 0.5
        if far_fraction < 0.1:
            fog_score = max(fog_score, 0.8)

        # Rain score: moderate density drop + higher intensity variance
        rain_score = 0.0
        if has_intensity:
            intensity = points[:, 3]
            intensity_std = float(np.std(intensity))
            intensity_mean = float(np.mean(intensity))

            # Coefficient of variation as a measure of intensity variability
            if intensity_mean > 0:
                cv = intensity_std / intensity_mean
            else:
                cv = 0.0

            # High intensity variance suggests wet reflections (rain)
            if cv > 0.5:
                rain_score = min((cv - 0.5) / 0.5, 1.0)

            # Also check if there's moderate density drop (less severe than fog)
            if 0.2 <= density_ratio < 0.5:
                rain_score = max(rain_score, 0.5)

        # Snow score: reduced density uniformly across all ranges + moderate variability
        snow_score = 0.0
        if total_points > 0:
            # Snow tends to reduce points uniformly and scatter in all directions
            expected_falloff = mid_density / near_density if near_density > 0 else 0.5
            # If falloff is gentle but overall density is low
            if 0.3 < expected_falloff < 0.8 and total_points < 50000:
                snow_score = 0.3
            if has_intensity:
                # Snow: relatively high mean intensity (reflective surface)
                if intensity_mean > 100:
                    snow_score += 0.3

        # Clear score: gradual density falloff, normal intensity behavior
        clear_score = 0.0
        if density_ratio >= self.fog_range_ratio_threshold:
            clear_score = density_ratio / 1.0  # approaches 1.0 when ratio is near 1
            clear_score = min(clear_score, 1.5)

        # Overcast is not readily detectable from lidar alone; assign small score
        overcast_score = 0.2

        scores: Dict[str, float] = {
            "clear": clear_score,
            "rain": rain_score,
            "fog": fog_score,
            "snow": snow_score,
            "overcast": overcast_score,
        }

        return _normalize_scores(scores)

    # ------------------------------------------------------------------
    # Radar-based classification
    # ------------------------------------------------------------------

    def classify_from_radar(self, radar_data: np.ndarray) -> Dict[str, float]:
        """Classify weather from radar detections.

        Analyzes clutter levels (low-RCS targets) and SNR characteristics to
        infer weather conditions.

        Parameters
        ----------
        radar_data : np.ndarray
            Nx5 (or more) array with columns [range, azimuth, doppler, rcs, snr].
            At minimum, columns at index 3 (RCS) and 4 (SNR) are used.

        Returns
        -------
        Dict[str, float]
            Normalized probability distribution over weather classes.
        """
        if radar_data.ndim != 2 or radar_data.shape[0] == 0:
            return {k: 1.0 / len(self.WEATHER_CLASSES) for k in self.WEATHER_CLASSES}

        n_targets = radar_data.shape[0]

        # Extract RCS and SNR columns
        rcs = radar_data[:, 3] if radar_data.shape[1] > 3 else np.zeros(n_targets)
        snr = radar_data[:, 4] if radar_data.shape[1] > 4 else np.full(n_targets, 20.0)

        # Clutter: targets with low RCS
        low_rcs_mask = rcs < self.radar_rcs_threshold
        clutter_count = int(np.sum(low_rcs_mask))
        clutter_ratio = clutter_count / n_targets if n_targets > 0 else 0.0

        # Mean SNR of clutter targets
        if clutter_count > 0:
            mean_clutter_snr = float(np.mean(snr[low_rcs_mask]))
        else:
            mean_clutter_snr = 20.0  # assume normal

        # Overall mean SNR
        mean_snr = float(np.mean(snr))

        # Scoring logic
        scores: Dict[str, float] = {k: 0.0 for k in self.WEATHER_CLASSES}

        # Clear: low clutter ratio and good SNR
        if clutter_ratio < 0.2:
            scores["clear"] = (1.0 - clutter_ratio) * min(mean_snr / 15.0, 1.5)

        # Rain: high clutter with low SNR suggests heavy rain
        if clutter_ratio > 0.3:
            rain_score = clutter_ratio
            if mean_snr < 10.0:
                rain_score *= 1.5  # heavy rain amplification
            scores["rain"] = min(rain_score, 2.0)

        # Fog: moderate-to-high clutter but with decent SNR (fog scatters but
        # does not heavily attenuate at automotive radar frequencies)
        if 0.2 <= clutter_ratio <= 0.6 and mean_snr >= 8.0:
            scores["fog"] = clutter_ratio * 0.8

        # Snow: moderate clutter, moderate SNR depression
        if 0.15 <= clutter_ratio <= 0.45 and 5.0 <= mean_snr <= 15.0:
            scores["snow"] = 0.5

        # Overcast: not strongly detectable from radar alone
        scores["overcast"] = 0.15

        return _normalize_scores(scores)

    # ------------------------------------------------------------------
    # Multi-sensor fusion
    # ------------------------------------------------------------------

    def fuse_classifications(
        self,
        camera_result: Optional[Dict[str, float]] = None,
        lidar_result: Optional[Dict[str, float]] = None,
        radar_result: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Fuse weather classifications from multiple sensors.

        Performs weighted averaging over available sensor results. Sensors that
        are unavailable (None) are excluded and their weight is redistributed
        proportionally among available sensors.

        Parameters
        ----------
        camera_result : Optional[Dict[str, float]]
            Classification output from classify_from_camera, or None if unavailable.
        lidar_result : Optional[Dict[str, float]]
            Classification output from classify_from_lidar, or None if unavailable.
        radar_result : Optional[Dict[str, float]]
            Classification output from classify_from_radar, or None if unavailable.

        Returns
        -------
        Dict[str, float]
            Normalized fused probability distribution over weather classes.

        Raises
        ------
        ValueError
            If all sensor inputs are None.
        """
        available: list = []
        weights: list = []

        if camera_result is not None:
            available.append(camera_result)
            weights.append(self.camera_weight)
        if lidar_result is not None:
            available.append(lidar_result)
            weights.append(self.lidar_weight)
        if radar_result is not None:
            available.append(radar_result)
            weights.append(self.radar_weight)

        if not available:
            raise ValueError(
                "At least one sensor classification must be provided for fusion."
            )

        # Normalize weights so they sum to 1.0
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Weighted average across all weather classes
        fused: Dict[str, float] = {k: 0.0 for k in self.WEATHER_CLASSES}

        for result, weight in zip(available, normalized_weights):
            for cls in self.WEATHER_CLASSES:
                fused[cls] += weight * result.get(cls, 0.0)

        return _normalize_scores(fused)
