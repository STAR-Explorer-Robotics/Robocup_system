"""QR detection baseline built on OpenCV QRCodeDetector."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from rescue_inspection.config import DetectorConfig, TargetDetectorConfig
from rescue_inspection.models import FramePacket, Point2D, QRDetectionResult, QRDetectionItem
from rescue_inspection.tasks.vision_target import TargetROIDetector, TargetROIDetectorDisabled
from rescue_inspection.tasks.vision_target.types import TargetROIDetection


class QRCodeBaselineDetector:
    """Run QR detection with curved-scene support and lightweight fallback preprocessing."""

    def __init__(
        self,
        config: DetectorConfig,
        target_detector_config: TargetDetectorConfig | None = None,
        *,
        logger: logging.Logger | None = None,
        roi_detector: TargetROIDetector | None = None,
    ) -> None:
        self.config = config
        self.target_detector_config = target_detector_config
        self.logger = logger or logging.getLogger(__name__)
        self._detector = cv2.QRCodeDetector()
        self._roi_detector = roi_detector
        self._roi_detector_disabled_reason: str | None = None

    def detect(self, packet: FramePacket) -> QRDetectionResult:
        """Detect and decode a QR code from a frame."""

        attempts: list[tuple[str, np.ndarray]] = [("standard", packet.image)]

        if self.config.try_curved:
            attempts.append(("curved", packet.image))

        if self.config.enable_preprocess_fallback:
            preprocessed = self._preprocess(packet.image)
            attempts.append(("fallback_standard", preprocessed))
            if self.config.try_curved:
                attempts.append(("fallback_curved", preprocessed))

        last_points: list[np.ndarray] = []
        failure_reason = "no_qr_candidate_detected"

        for attempt_name, image in attempts:
            decoded_texts, points_list, error_reason = self._decode_with_method(attempt_name, image)
            if points_list:
                last_points = points_list

            valid_texts = [t.strip() for t in decoded_texts if t.strip()]

            if valid_texts:
                method_used = "fallback" if attempt_name.startswith("fallback") else attempt_name
                items = []
                for text, pts in zip(decoded_texts, points_list):
                    items.append(QRDetectionItem(
                        decoded_text=text.strip(),
                        polygon=self._points_to_polygon(pts),
                        is_suspect=not text.strip(),
                        failure_reason=None if text.strip() else f"decode_failed_after_{attempt_name}",
                    ))
                
                primary_item = self._select_primary_qr(items)

                return QRDetectionResult(
                    timestamp=packet.timestamp,
                    frame_id=packet.frame_id,
                    detected=True,
                    decoded_text=primary_item.decoded_text,
                    polygon=primary_item.polygon,
                    items=items,
                    method_used=method_used,
                    source_type=packet.source_type,
                )

            if error_reason is not None:
                failure_reason = error_reason
                continue

            if points_list and self.config.enable_candidate_decode_fallback:
                items = []
                any_decoded = False
                for pts in points_list:
                    candidate_text = self._decode_from_candidate(packet.image, image, pts)
                    items.append(QRDetectionItem(
                        decoded_text=candidate_text,
                        polygon=self._points_to_polygon(pts),
                        is_suspect=not candidate_text,
                        failure_reason=None if candidate_text else f"decode_failed_after_{attempt_name}",
                    ))
                    if candidate_text:
                        any_decoded = True
                        
                if any_decoded:
                    primary_item = self._select_primary_qr(items)
                    return QRDetectionResult(
                        timestamp=packet.timestamp,
                        frame_id=packet.frame_id,
                        detected=True,
                        decoded_text=primary_item.decoded_text,
                        polygon=primary_item.polygon,
                        items=items,
                        method_used="fallback",
                        source_type=packet.source_type,
                    )

            if points_list:
                failure_reason = f"decode_failed_after_{attempt_name}"

        items = []
        for pts in last_points:
            items.append(
                QRDetectionItem(
                    decoded_text="",
                    polygon=self._points_to_polygon(pts),
                    is_suspect=True,
                    failure_reason=failure_reason,
                )
            )
            
        primary_item = self._select_primary_qr(items) if items else None
        roi_result = self._try_roi_decode(packet, failure_reason=failure_reason)
        if roi_result is not None:
            return roi_result

        return QRDetectionResult(
            timestamp=packet.timestamp,
            frame_id=packet.frame_id,
            detected=False,
            decoded_text="",
            polygon=primary_item.polygon if primary_item else [],
            items=items,
            method_used="none",
            source_type=packet.source_type,
            failure_reason=failure_reason,
        )

    def _try_roi_decode(self, packet: FramePacket, *, failure_reason: str) -> QRDetectionResult | None:
        """Try DL ROI localization after full-frame QR decode failed."""

        roi_detector = self._get_roi_detector()
        if roi_detector is None:
            return None

        detections = roi_detector.detect(packet, class_names={"qr_roi"})
        if not detections:
            return None

        items: list[QRDetectionItem] = []
        diagnostics: dict[str, object] = {
            "roi_candidate_count": len(detections),
            "roi_detector_status": "attempted",
            "full_frame_failure_reason": failure_reason,
        }
        for index, detection in enumerate(detections):
            variant_result = self._decode_roi_variants(packet.image, detection)
            text, variant, variant_diagnostics = _normalize_roi_decode_result(variant_result)
            diagnostics[f"candidate_{index}_variant"] = variant or "decode_failed"
            for key, value in variant_diagnostics.items():
                diagnostics[f"candidate_{index}_{key}"] = value
            if not text:
                diagnostics[f"candidate_{index}_failure_reason"] = "qr_roi_decode_failed"
            items.append(
                QRDetectionItem(
                    decoded_text=text,
                    polygon=detection.polygon,
                    is_suspect=not text,
                    failure_reason=None if text else "qr_roi_decode_failed",
                )
            )
            if text:
                return QRDetectionResult(
                    timestamp=packet.timestamp,
                    frame_id=packet.frame_id,
                    detected=True,
                    decoded_text=text,
                    polygon=detection.polygon,
                    items=items,
                    method_used=f"roi_{variant}",
                    source_type=packet.source_type,
                    roi_candidate_count=len(detections),
                    roi_diagnostics=diagnostics,
                )

        primary_item = self._select_primary_qr(items)
        return QRDetectionResult(
            timestamp=packet.timestamp,
            frame_id=packet.frame_id,
            detected=False,
            decoded_text="",
            polygon=primary_item.polygon,
            items=items,
            method_used="roi_decode_failed",
            source_type=packet.source_type,
            failure_reason="qr_roi_detected_decode_failed",
            roi_candidate_count=len(detections),
            roi_diagnostics=diagnostics,
        )

    def _get_roi_detector(self) -> TargetROIDetector | None:
        if self._roi_detector is not None:
            return self._roi_detector
        if self._roi_detector_disabled_reason is not None or self.target_detector_config is None:
            return None
        if not self.target_detector_config.enabled:
            self._roi_detector_disabled_reason = "target_detector.enabled is false"
            return None
        try:
            self._roi_detector = TargetROIDetector(self.target_detector_config, logger=self.logger)
        except TargetROIDetectorDisabled as exc:
            self._roi_detector_disabled_reason = str(exc)
            return None
        return self._roi_detector

    def _decode_roi_variants(
        self,
        original_image: np.ndarray,
        detection: TargetROIDetection,
    ) -> tuple[str, str, dict[str, object]]:
        """Try QR decode variants for one ROI crop."""

        variant_images: list[tuple[str, np.ndarray, dict[str, object]]] = []
        expanded_bbox = _expanded_bbox_for_detection(
            original_image,
            detection,
            pad_px=self.config.qr_roi_decode_pad_px,
            pad_ratio=self.config.qr_roi_decode_pad_ratio,
        )
        if detection.crop is not None:
            variant_images.append(("roi_original", detection.crop, {}))
            bordered = _add_white_border(detection.crop, self.config.qr_roi_decode_white_border_px)
            if bordered.size:
                variant_images.append(("roi_with_white_border", bordered, {"used_white_border": True}))
                scaled = _scale_roi(bordered, self.config.qr_roi_decode_scale)
                if scaled.size:
                    variant_images.append(
                        (
                            "roi_scaled_with_white_border",
                            scaled,
                            {"used_white_border": True, "used_scale": True},
                        )
                    )
        expanded_crop = _crop_with_padding(
            original_image,
            detection,
            pad_px=self.config.qr_roi_decode_pad_px,
            pad_ratio=self.config.qr_roi_decode_pad_ratio,
        )
        if expanded_crop is not None and expanded_bbox is not None:
            expanded_diag: dict[str, object] = {
                "used_expanded_crop": True,
                "expanded_bbox": list(expanded_bbox),
            }
            variant_images.append(("roi_expanded", expanded_crop, expanded_diag))
            bordered = _add_white_border(expanded_crop, self.config.qr_roi_decode_white_border_px)
            if bordered.size:
                variant_images.append(
                    (
                        "roi_expanded_white_border",
                        bordered,
                        {**expanded_diag, "used_white_border": True},
                    )
                )
                scaled = _scale_roi(bordered, self.config.qr_roi_decode_scale)
                if scaled.size:
                    variant_images.append(
                        (
                            "roi_expanded_scaled_white_border",
                            scaled,
                            {**expanded_diag, "used_white_border": True, "used_scale": True},
                        )
                    )
        if detection.square_crop is not None:
            variant_images.append(("roi_square", detection.square_crop, {}))
            bordered = _add_white_border(detection.square_crop, self.config.qr_roi_decode_white_border_px)
            if bordered.size:
                variant_images.append(("roi_square_white_border", bordered, {"used_white_border": True}))
        if detection.crop is not None:
            enlarged = cv2.resize(detection.crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            variant_images.append(("roi_enlarged", enlarged, {"used_legacy_enlarged": True}))
        if detection.polygon:
            points = np.array([[p.x, p.y] for p in detection.polygon], dtype=np.float32).reshape(1, -1, 2)
            warped = self._warp_from_points(original_image, points)
            variant_images.append(("roi_perspective", warped, {}))
            bordered = _add_white_border(warped, self.config.qr_roi_decode_white_border_px)
            if bordered.size:
                variant_images.append(("roi_perspective_white_border", bordered, {"used_white_border": True}))

        enriched: list[tuple[str, np.ndarray, dict[str, object]]] = []
        for name, image, diagnostics in variant_images:
            if image.size == 0:
                continue
            enriched.append((name, image, diagnostics))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            enriched.append((f"{name}_gray", gray, diagnostics))
            enriched.append((f"{name}_clahe", self._preprocess(image), diagnostics))
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            enriched.append((f"{name}_sharpen", cv2.filter2D(image, ddepth=-1, kernel=kernel), diagnostics))
            adaptive = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                3,
            )
            enriched.append((f"{name}_adaptive_threshold", adaptive, diagnostics))

        failure_diagnostics: dict[str, object] = {}
        if expanded_bbox is not None:
            failure_diagnostics["expanded_bbox"] = list(expanded_bbox)
        for name, image, diagnostics in enriched:
            texts, _, _ = self._decode_with_method("standard", image)
            valid_texts = [text.strip() for text in texts if text.strip()]
            if valid_texts:
                return valid_texts[0], name, diagnostics
            if self.config.try_curved:
                texts, _, _ = self._decode_with_method("curved", image)
                valid_texts = [text.strip() for text in texts if text.strip()]
                if valid_texts:
                    return valid_texts[0], name, diagnostics
        return "", "", failure_diagnostics

    def _decode_with_method(
        self, method_name: str, image: np.ndarray
    ) -> tuple[list[str], list[np.ndarray], str | None]:
        try:
            decoded_texts = []
            all_points = []
            
            if method_name.endswith("curved"):
                if not hasattr(self._detector, "detectAndDecodeMultiCurved"):
                    if not hasattr(self._detector, "detectAndDecodeCurved"):
                        return [], [], None
                    text, pts, _ = self._detector.detectAndDecodeCurved(image)
                    if pts is not None:
                        return [text] if text else [""], [pts], None
                    return [], [], None
                
                retval, decoded_info, points, _ = self._detector.detectAndDecodeMultiCurved(image)
            else:
                retval, decoded_info, points, _ = self._detector.detectAndDecodeMulti(image)
                
            if retval and points is not None:
                for i in range(len(points)):
                    text = decoded_info[i] if i < len(decoded_info) else ""
                    decoded_texts.append(text)
                    all_points.append(points[i])
                return decoded_texts, all_points, None
            
            # fallback to single
            if method_name.endswith("curved"):
                text, pts, _ = self._detector.detectAndDecodeCurved(image)
            else:
                text, pts, _ = self._detector.detectAndDecode(image)
                
            if pts is not None:
                return [text] if text else [""], [pts], None
                
            return [], [], None
        except cv2.error:
            return [], [], f"opencv_error_after_{method_name}"

    def _decode_from_candidate(
        self,
        original_image: np.ndarray,
        working_image: np.ndarray,
        points: np.ndarray,
    ) -> str:
        """Try extra decode paths once a QR quadrilateral has already been detected."""

        candidate_images: list[np.ndarray] = [working_image]
        if not np.shares_memory(working_image, original_image):
            candidate_images.append(original_image)

        for candidate_image in candidate_images:
            sanitized_points = self._sanitize_points(points, candidate_image)

            decoded_text = self._decode_with_points(candidate_image, sanitized_points, curved=False)
            if decoded_text:
                return decoded_text

            if self.config.try_curved:
                decoded_text = self._decode_with_points(candidate_image, sanitized_points, curved=True)
                if decoded_text:
                    return decoded_text

            warped = self._warp_from_points(candidate_image, sanitized_points)
            texts, _, _ = self._decode_with_method("standard", warped)
            if texts and texts[0]:
                return texts[0].strip()

            if self.config.try_curved:
                texts, _, _ = self._decode_with_method("curved", warped)
                if texts and texts[0]:
                    return texts[0].strip()

            if self.config.enable_preprocess_fallback:
                warped_preprocessed = self._preprocess(warped)
                texts, _, _ = self._decode_with_method("fallback_standard", warped_preprocessed)
                if texts and texts[0]:
                    return texts[0].strip()

                if self.config.try_curved:
                    texts, _, _ = self._decode_with_method("fallback_curved", warped_preprocessed)
                    if texts and texts[0]:
                        return texts[0].strip()

        return ""

    def _decode_with_points(self, image: np.ndarray, points: np.ndarray, *, curved: bool) -> str:
        """Decode from an already-detected quadrilateral."""

        try:
            if curved:
                if not hasattr(self._detector, "decodeCurved"):
                    return ""
                decoded_text, _ = self._detector.decodeCurved(image, points)
                return decoded_text.strip()

            decoded_text, _ = self._detector.decode(image, points)
            return decoded_text.strip()
        except cv2.error:
            return ""

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply a light grayscale-contrast-sharpen fallback."""

        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(enhanced, ddepth=-1, kernel=kernel)

    def _sanitize_points(self, points: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Clip detected points to valid image bounds before using them again."""

        reshaped = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        height, width = image.shape[:2]
        reshaped[:, 0] = np.clip(reshaped[:, 0], 0.0, float(max(width - 1, 0)))
        reshaped[:, 1] = np.clip(reshaped[:, 1], 0.0, float(max(height - 1, 0)))
        return reshaped.reshape(1, -1, 2)

    def _warp_from_points(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Rectify a detected QR quadrilateral into a square patch."""

        ordered = self._order_points(points)
        target_size = self.config.warped_patch_size
        destination = np.array(
            [
                [0.0, 0.0],
                [float(target_size - 1), 0.0],
                [float(target_size - 1), float(target_size - 1)],
                [0.0, float(target_size - 1)],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(ordered, destination)
        return cv2.warpPerspective(image, matrix, (target_size, target_size))

    def _order_points(self, points: np.ndarray) -> np.ndarray:
        """Order points as top-left, top-right, bottom-right, bottom-left."""

        reshaped = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        sums = reshaped.sum(axis=1)
        diffs = np.diff(reshaped, axis=1).reshape(-1)

        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = reshaped[np.argmin(sums)]
        ordered[2] = reshaped[np.argmax(sums)]
        ordered[1] = reshaped[np.argmin(diffs)]
        ordered[3] = reshaped[np.argmax(diffs)]
        return ordered

    def _points_to_polygon(self, points: np.ndarray | None) -> list[Point2D]:
        if points is None:
            return []

        reshaped = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        return [Point2D(x=float(x), y=float(y)) for x, y in reshaped]

    def _select_primary_qr(self, items: list[QRDetectionItem]) -> QRDetectionItem:
        if not items:
            return QRDetectionItem()
            
        def polygon_area(poly: list[Point2D]) -> float:
            if not poly or len(poly) < 3:
                return 0.0
            pts = np.array([[p.x, p.y] for p in poly], dtype=np.float32)
            return cv2.contourArea(pts)

        with_text = [item for item in items if item.decoded_text]
        if with_text:
            return max(with_text, key=lambda item: polygon_area(item.polygon))
            
        return max(items, key=lambda item: polygon_area(item.polygon))


def _normalize_roi_decode_result(result: tuple[Any, ...]) -> tuple[str, str, dict[str, object]]:
    """Accept old two-field test doubles and the current diagnostics-aware result."""

    if len(result) == 2:
        text, variant = result
        return str(text or ""), str(variant or ""), {}
    text, variant, diagnostics = result
    return str(text or ""), str(variant or ""), dict(diagnostics or {})


def _expanded_bbox_for_detection(
    image: np.ndarray,
    detection: TargetROIDetection,
    *,
    pad_px: int,
    pad_ratio: float,
) -> tuple[int, int, int, int] | None:
    """Return an expanded ROI bbox clipped to image bounds."""

    bbox = _bbox_from_detection(detection)
    if bbox is None:
        return None
    return _expand_bbox(bbox, image.shape, pad_px=pad_px, pad_ratio=pad_ratio)


def _bbox_from_detection(detection: TargetROIDetection) -> tuple[float, float, float, float] | None:
    """Prefer detector bbox, falling back to the polygon bounds."""

    bbox = detection.bbox_xyxy
    if _valid_bbox(bbox):
        return bbox
    return _bbox_from_polygon(detection.polygon)


def _bbox_from_polygon(polygon: list[Point2D]) -> tuple[float, float, float, float] | None:
    """Return xyxy bounds for a polygon."""

    if not polygon:
        return None
    xs = [point.x for point in polygon]
    ys = [point.y for point in polygon]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return bbox if _valid_bbox(bbox) else None


def _valid_bbox(bbox: tuple[float, float, float, float] | None) -> bool:
    """Return true when a bbox has positive extent."""

    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    return np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1


def _expand_bbox(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
    *,
    pad_px: int,
    pad_ratio: float,
) -> tuple[int, int, int, int] | None:
    """Expand a bbox and clip it to image bounds for array slicing."""

    if not _valid_bbox(bbox) or len(image_shape) < 2:
        return None
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        return None
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    pad = max(float(pad_px), float(max(box_w, box_h) * pad_ratio))
    ix1 = max(0, int(np.floor(x1 - pad)))
    iy1 = max(0, int(np.floor(y1 - pad)))
    ix2 = min(width, int(np.ceil(x2 + pad)))
    iy2 = min(height, int(np.ceil(y2 + pad)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def _crop_with_padding(
    image: np.ndarray,
    detection: TargetROIDetection,
    *,
    pad_px: int,
    pad_ratio: float,
) -> np.ndarray | None:
    """Crop an expanded ROI from the original frame."""

    expanded = _expanded_bbox_for_detection(image, detection, pad_px=pad_px, pad_ratio=pad_ratio)
    if expanded is None:
        return None
    x1, y1, x2, y2 = expanded
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop.copy()


def _add_white_border(image: np.ndarray, border_px: int) -> np.ndarray:
    """Pad a gray or color ROI with a white quiet-zone border."""

    if image.size == 0:
        return image.copy()
    if border_px <= 0:
        return image.copy()
    if image.ndim == 2:
        value: int | tuple[int, ...] = 255
    elif image.ndim == 3:
        value = tuple(255 for _ in range(image.shape[2]))
    else:
        return image.copy()
    return cv2.copyMakeBorder(
        image,
        border_px,
        border_px,
        border_px,
        border_px,
        borderType=cv2.BORDER_CONSTANT,
        value=value,
    )


def _scale_roi(image: np.ndarray, scale: float) -> np.ndarray:
    """Scale an ROI for QR decoding while keeping invalid images safe."""

    if image.size == 0:
        return image.copy()
    if scale <= 1.0:
        return image.copy()
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
