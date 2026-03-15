import re
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# ─── Known-text dictionary ──────────────────────────────────────────

_KNOWN_TEXTS_CACHE = None

def load_known_texts(
    annotations_dir="/Volumes/ExternalSSD/Projects/PillOCR/master_dataset_v5/merged/annotations",
):
    """Build a set of known upper-case pill texts from the merged annotations.

    Returns a dict  { normalised_text: count }  so callers can also
    rank by frequency.
    """
    global _KNOWN_TEXTS_CACHE
    if _KNOWN_TEXTS_CACHE is not None:
        return _KNOWN_TEXTS_CACHE

    counter: Counter = Counter()
    ann_path = Path(annotations_dir)
    if ann_path.is_dir():
        for fn in ann_path.iterdir():
            if fn.suffix != ".json":
                continue
            with open(fn) as f:
                data = json.load(f)
            for r in data.get("regions", []):
                t = r.get("text", "").strip().upper()
                if t:
                    counter[t] += 1

    _KNOWN_TEXTS_CACHE = dict(counter)
    return _KNOWN_TEXTS_CACHE


def _normalise(text: str) -> str:
    """Strip and upper-case for matching."""
    return text.strip().upper()


def _match_score(ocr_text: str) -> tuple:
    """Score an OCR result for tiebreaking within majority vote.

    Returns a 3-tuple for comparison:
        (has_text: bool, is_numeric: bool, text_length: int)

    Only used to pick the best orientation *within* the winning text group.
    The actual winner is decided by vote count (majority), not this score.
    """
    norm = ocr_text.strip().upper()
    if not norm:
        return (False, False, 0)
    is_numeric = bool(re.fullmatch(r'[\d.,/ ]+', norm))  # e.g. "500", "10/20"
    return (True, is_numeric, len(norm))


# All geometry variants: 4 rotations × 3 flip modes = 12 combos
_ROTATIONS = [0, 90, 180, 270]
_FLIPS = ["None", "Horizontal", "Vertical"]


def brute_force_best_orientation(
    crop_np,
    processor,
    model,
    device,
    known_texts=None,  # kept for backward compat but unused in scoring
):
    """Try every rotation+flip combo, OCR each with both methods,
    and pick the text that appears most often (majority vote).

    Returns a dict:
        rotation, flip          – winning geometry
        best_text               – best OCR text found
        best_method             – '1' or '2'
        is_numeric              – True if text is purely digits
        vote_count              – how many attempts produced this text
        total_attempts          – total number of attempts (24)
        all_results             – list of all 24 (rot, flip, method, text) entries
        best_crop_np            – the transformed crop that won
        best_processed_img      – the preprocessed image that won
    """
    all_results = []

    for rot in _ROTATIONS:
        for flip in _FLIPS:
            transformed = apply_crop_transform(crop_np, rot, flip)

            # --- Method 1 ---
            proc1 = process_adaptive_inverted_closing(transformed)
            text1 = predict_text(proc1, processor, model, device)
            score1 = _match_score(text1)
            all_results.append({
                "rotation": rot, "flip": flip, "method": "1",
                "text": text1, "score": score1,
                "crop_np": transformed, "processed_img": proc1,
            })

            # --- Method 2 ---
            proc2 = process_clahe_adaptive(transformed)
            text2 = predict_text(proc2, processor, model, device)
            score2 = _match_score(text2)
            all_results.append({
                "rotation": rot, "flip": flip, "method": "2",
                "text": text2, "score": score2,
                "crop_np": transformed, "processed_img": proc2,
            })

    # ── Majority vote: pick the text that appears most often ──
    text_counts = Counter()
    text_entries = {}  # normalised text → list of entries
    for e in all_results:
        norm = e["text"].strip().upper()
        if not norm:
            continue
        text_counts[norm] += 1
        text_entries.setdefault(norm, []).append(e)

    if text_counts:
        # Primary: vote count.  Tiebreak: is_numeric > longer text
        def _candidate_key(norm_text):
            count = text_counts[norm_text]
            best_entry = max(text_entries[norm_text], key=lambda e: e["score"])
            return (count, best_entry["score"])

        winner_text = max(text_counts, key=_candidate_key)
        info = max(text_entries[winner_text], key=lambda e: e["score"])
        vote_count = text_counts[winner_text]
    else:
        info = all_results[0]
        vote_count = 0

    is_numeric = info["score"][1]
    return {
        "rotation": info["rotation"],
        "flip": info["flip"],
        "best_text": info["text"],
        "best_method": info["method"],
        "matched": False,  # kept for backward compat; unused now
        "is_numeric": is_numeric,
        "vote_count": vote_count,
        "total_attempts": len(all_results),
        "all_results": [
            (e["rotation"], e["flip"], e["method"], e["text"])
            for e in all_results
        ],
        "best_crop_np": info["crop_np"],
        "best_processed_img": info["processed_img"],
    }

def load_trocr_model(model_path, device):
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    return processor, model

def process_adaptive_inverted_closing(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    inverted = cv2.bitwise_not(adaptive_thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=1)
    return Image.fromarray(closed).convert("RGB")

def process_clahe_adaptive(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    adaptive_thresh = cv2.adaptiveThreshold(
        clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    return Image.fromarray(adaptive_thresh).convert("RGB")

def predict_text(image_rgb, processor, model, device):
    pixel_values = processor(image_rgb, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_new_tokens=30)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def apply_crop_transform(crop_np, rotation=0, flip="None"):
    """Apply rotation and flip to a crop image (numpy BGR array).

    Args:
        crop_np: numpy array (H, W, C)
        rotation: one of 0, 90, 180, 270 (degrees counter-clockwise)
        flip: one of 'None', 'Horizontal', 'Vertical'

    Returns:
        Transformed numpy array
    """
    img = crop_np.copy()

    if rotation == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotation == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if flip == "Horizontal":
        img = cv2.flip(img, 1)
    elif flip == "Vertical":
        img = cv2.flip(img, 0)

    return img


def is_obb_model(yolo_model):
    """Check if the YOLO model is an OBB (oriented bounding box) model."""
    task = getattr(yolo_model, 'task', None)
    if task == 'obb':
        return True
    # Fallback: check model name/type string
    model_name = str(getattr(yolo_model, 'model_name', ''))
    cfg = str(getattr(yolo_model, 'cfg', ''))
    return 'obb' in model_name.lower() or 'obb' in cfg.lower()


def crop_obb_region(image, corners, pad=10):
    """Crop a rotated bounding box region from an image, returning an
    axis-aligned image of the content inside the OBB.

    Args:
        image: numpy array (H, W, C)
        corners: 4 corner points as numpy array shape (4, 2), ordered
        pad: number of pixels to expand the OBB outward on each side

    Returns:
        Cropped and de-rotated image (numpy array)
    """
    pts = np.array(corners, dtype=np.float32)

    # Expand corners outward from the centroid by `pad` pixels
    if pad > 0:
        centroid = pts.mean(axis=0)
        directions = pts - centroid
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-6, None)  # avoid division by zero
        pts = pts + (directions / norms) * pad
        # Clamp to image bounds
        h_img, w_img = image.shape[:2]
        pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)

    # Compute width and height of the (expanded) oriented box
    w1 = np.linalg.norm(pts[1] - pts[0])
    w2 = np.linalg.norm(pts[2] - pts[3])
    h1 = np.linalg.norm(pts[3] - pts[0])
    h2 = np.linalg.norm(pts[2] - pts[1])

    width = int(max(w1, w2))
    height = int(max(h1, h2))

    if width < 1 or height < 1:
        return None

    # Destination points for perspective transform
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst_pts)
    cropped = cv2.warpPerspective(image, M, (width, height))
    return cropped


def draw_obb_boxes(image, results, class_names, colors, show_label=False, original_image=None):
    """Draw axis-aligned bounding boxes (for regular YOLO detect models).
    
    Crops are taken from `original_image` (the clean, un-annotated image)
    so that drawn boxes/labels never leak into OCR inputs.
    """
    if original_image is None:
        original_image = image  # fallback: old behaviour
    object_counts = {}
    crops_info = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = class_names[cls_id]
                
                if class_name not in object_counts:
                    object_counts[class_name] = 0
                object_counts[class_name] += 1
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                h, w = image.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if (y2 > y1) and (x2 > x1):
                    crop_img = original_image[y1:y2, x1:x2].copy()
                    crops_info.append({
                        'crop': crop_img,
                        'bbox': (x1, y1, x2, y2),
                        'corners': None,  # not OBB
                        'class_name': class_name,
                        'confidence': confidence
                    })

                # Draw box on display copy (skip pill class)
                if cls_id != 0:
                    color = colors.get(class_name, (255, 255, 255))
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    if show_label:
                        label = f"{class_name}: {confidence*100:.0f}%"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                                     (x1 + label_size[0], y1), color, -1)
                        cv2.putText(image, label, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return image, object_counts, crops_info


def draw_obb_oriented_boxes(image, results, class_names, colors, show_label=False, original_image=None):
    """Draw oriented bounding boxes (for YOLO OBB models).

    Uses result.obb which provides xyxyxyxy (4 corner points) for each
    detection. Crops are de-rotated using perspective transform.

    Crops are taken from `original_image` (the clean, un-annotated image)
    so that drawn boxes/labels never leak into OCR inputs.
    """
    if original_image is None:
        original_image = image  # fallback: old behaviour
    object_counts = {}
    crops_info = []

    for result in results:
        obbs = result.obb
        if obbs is None:
            continue

        for i in range(len(obbs)):
            cls_id = int(obbs.cls[i])
            confidence = float(obbs.conf[i])
            class_name = class_names[cls_id]

            if class_name not in object_counts:
                object_counts[class_name] = 0
            object_counts[class_name] += 1

            # xyxyxyxy gives 4 corner points as (x1,y1, x2,y2, x3,y3, x4,y4)
            corners = obbs.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
            corners_int = corners.astype(np.int32)

            # Crop the rotated region from the CLEAN original image
            crop_img = crop_obb_region(original_image, corners)
            if crop_img is not None and crop_img.size > 0:
                # Also compute axis-aligned bounding rect for label placement
                x_min = int(corners[:, 0].min())
                y_min = int(corners[:, 1].min())
                x_max = int(corners[:, 0].max())
                y_max = int(corners[:, 1].max())

                crops_info.append({
                    'crop': crop_img,
                    'bbox': (x_min, y_min, x_max, y_max),
                    'corners': corners_int,
                    'class_name': class_name,
                    'confidence': confidence
                })

            # Draw the oriented polygon on the display copy (skip pill class)
            if cls_id != 0:
                color = colors.get(class_name, (255, 255, 255))
                cv2.polylines(image, [corners_int], isClosed=True, color=color, thickness=2)

                if show_label:
                    # Place label at the top-most corner
                    top_idx = np.argmin(corners[:, 1])
                    lx, ly = int(corners[top_idx, 0]), int(corners[top_idx, 1])
                    label = f"{class_name}: {confidence*100:.0f}%"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(image, (lx, ly - label_size[1] - 10),
                                  (lx + label_size[0], ly), color, -1)
                    cv2.putText(image, label, (lx, ly - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return image, object_counts, crops_info


def display_counts(image, object_counts, colors=None):
    y_offset = 30
    total_objects = sum(object_counts.values())
    cv2.putText(image, f"Total Objects: {total_objects}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30
    for class_name, count in object_counts.items():
        text = f"{class_name}: {count}"
        color = colors.get(class_name, (255, 255, 255)) if colors else (255, 255, 255)
        cv2.putText(image, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25
    return image


# ─── Text-zone-inside-pill filtering ────────────────────────────────

def _bbox_iou(box_a, box_b):
    """Compute IoU between two axis-aligned bounding boxes.
    
    Each box is (x1, y1, x2, y2).
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _text_zone_inside_pill(tz_bbox, pill_bboxes, containment_thresh=0.30):
    """Return True if the text_zone is sufficiently inside any pill.
    
    Instead of plain IoU (which penalises size mismatch), we use a
    *containment ratio*: intersection_area / text_zone_area.
    A text_zone that is fully inside a pill scores 1.0; one that only
    partially overlaps scores proportionally.  Threshold defaults to 30%
    so that even partially visible text on a pill's edge is kept.
    """
    tx1, ty1, tx2, ty2 = tz_bbox
    tz_area = (tx2 - tx1) * (ty2 - ty1)
    if tz_area <= 0:
        return False
    
    for px1, py1, px2, py2 in pill_bboxes:
        # Intersection
        ix1 = max(tx1, px1)
        iy1 = max(ty1, py1)
        ix2 = min(tx2, px2)
        iy2 = min(ty2, py2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        
        containment = inter / tz_area
        if containment >= containment_thresh:
            return True
    
    return False

def predict_pill_ocr(yolo_model, trocr_processor, trocr_model, source_path, trocr_device,
                     save_dir_base="outputs", conf=0.5, iou_threshold=0.45,
                     crop_rotation=0, crop_flip="None",
                     auto_orient=False):
    
    save_dir = os.path.join(save_dir_base, "images")
    os.makedirs(save_dir, exist_ok=True)

    image = cv2.imread(source_path)
    if image is None:
        raise ValueError(f"Could not load image: {source_path}")

    class_names = yolo_model.names
    colors = {name: tuple(np.random.randint(0, 255, 3).tolist()) for name in class_names.values()}

    # Determine if this is an OBB model
    obb_mode = is_obb_model(yolo_model)

    # Stage 1: YOLO Detection
    results = yolo_model.predict(
        source=source_path,
        save=False,
        conf=conf,
        iou=iou_threshold,
        device=yolo_model.device,
    )

    # Draw boxes based on model type
    # Pass the original image separately so crops come from the clean version
    if obb_mode:
        image_with_boxes, object_counts, crops_info = draw_obb_oriented_boxes(
            image.copy(), results, class_names, colors, show_label=True,
            original_image=image
        )
    else:
        image_with_boxes, object_counts, crops_info = draw_obb_boxes(
            image.copy(), results, class_names, colors, show_label=True,
            original_image=image
        )
    
    final_image = display_counts(image_with_boxes, object_counts, colors=colors)
    
    # ── Filter text_zones: only keep those inside a pill ──
    pill_bboxes = [c['bbox'] for c in crops_info if c['class_name'] == 'pill']
    filtered_crops = []
    for c in crops_info:
        if c['class_name'] == 'text_zone':
            # Check if this text_zone overlaps any pill bbox
            if not pill_bboxes or not _text_zone_inside_pill(c['bbox'], pill_bboxes):
                continue  # discard – not inside any pill
        filtered_crops.append(c)
    crops_info = filtered_crops
    
    pill_results = []
    
    # Stage 2 + 3: Only run TrOCR on text_zone crops (not pill crops)
    for idx, crop_data in enumerate(crops_info):
        # Skip non-text detections — only OCR the text_zone class
        if crop_data['class_name'] != 'text_zone':
            continue

        if auto_orient:
            # ── Brute-force all 12 geometry combos ──
            bf = brute_force_best_orientation(
                crop_data['crop'], trocr_processor, trocr_model,
                trocr_device
            )
            crop_np = bf["best_crop_np"]

            # Run both methods on the winning orientation so the user sees both
            processed_img_1 = process_adaptive_inverted_closing(crop_np)
            text_1 = predict_text(processed_img_1, trocr_processor, trocr_model, trocr_device)

            processed_img_2 = process_clahe_adaptive(crop_np)
            text_2 = predict_text(processed_img_2, trocr_processor, trocr_model, trocr_device)

            # Draw text label on the final image
            x1, y1, x2, y2 = crop_data['bbox']
            match_tag = "✓" if bf["matched"] else "?"
            label_text = f"{match_tag} {bf['best_text']} (r{bf['rotation']}° {bf['flip']} M{bf['best_method']})"
            color = colors.get(crop_data['class_name'], (255, 255, 255))
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(final_image, (x1, y2),
                         (x1 + label_size[0], y2 + label_size[1] + 10), color, -1)
            cv2.putText(final_image, label_text, (x1, y2 + label_size[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            pill_results.append({
                'original_crop': Image.fromarray(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB)),
                'processed_1': processed_img_1,
                'text_1': text_1,
                'processed_2': processed_img_2,
                'text_2': text_2,
                'auto_orient': bf,
            })
        else:
            # ── Manual orientation ──
            crop_np = apply_crop_transform(crop_data['crop'], crop_rotation, crop_flip)
            
            # Method 1
            processed_img_1 = process_adaptive_inverted_closing(crop_np)
            text_1 = predict_text(processed_img_1, trocr_processor, trocr_model, trocr_device)
            
            # Method 2
            processed_img_2 = process_clahe_adaptive(crop_np)
            text_2 = predict_text(processed_img_2, trocr_processor, trocr_model, trocr_device)
            
            # Draw text label on the final image
            x1, y1, x2, y2 = crop_data['bbox']
            label_text = f"M1: {text_1} | M2: {text_2}"
            color = colors.get(crop_data['class_name'], (255,255,255))
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(final_image, (x1, y2), 
                         (x1 + label_size[0], y2 + label_size[1] + 10), color, -1)
            cv2.putText(final_image, label_text, (x1, y2 + label_size[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                       
            pill_results.append({
                'original_crop': Image.fromarray(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB)),
                'processed_1': processed_img_1,
                'text_1': text_1,
                'processed_2': processed_img_2,
                'text_2': text_2,
                'auto_orient': None,
            })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(save_dir, f"{Path(source_path).stem}_{timestamp}.jpg")
    cv2.imwrite(output_path, final_image)
    
    return final_image, output_path, pill_results

def predict_cropped_pill_ocr(trocr_processor, trocr_model, source_path, trocr_device,
                             auto_orient=False):
    image = cv2.imread(source_path)
    if image is None:
        raise ValueError(f"Could not load image: {source_path}")
        
    crop_np = image
    pill_results = []

    if auto_orient:
        bf = brute_force_best_orientation(
            crop_np, trocr_processor, trocr_model, trocr_device
        )
        pill_results.append({
            'original_crop': Image.fromarray(
                cv2.cvtColor(bf["best_crop_np"], cv2.COLOR_BGR2RGB)
            ),
            'processed_1': bf["best_processed_img"],
            'text_1': bf["best_text"],
            'processed_2': bf["best_processed_img"],
            'text_2': bf["best_text"],
            'auto_orient': bf,
        })
    else:
        # Method 1
        processed_img_1 = process_adaptive_inverted_closing(crop_np)
        text_1 = predict_text(processed_img_1, trocr_processor, trocr_model, trocr_device)
        
        # Method 2
        processed_img_2 = process_clahe_adaptive(crop_np)
        text_2 = predict_text(processed_img_2, trocr_processor, trocr_model, trocr_device)
        
        pill_results.append({
            'original_crop': Image.fromarray(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB)),
            'processed_1': processed_img_1,
            'text_1': text_1,
            'processed_2': processed_img_2,
            'text_2': text_2,
            'auto_orient': None,
        })
    
    return pill_results