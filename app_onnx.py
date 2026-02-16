# Author: Clarence Jay Fetalino
# this scripts is designed to run on streamlit
# it has the capability to ingest one or multiple images, and output the confidence levels of vehicles detected
# users can visualize the data in tables, and can download them in csv format
#
# Update:
# - Adds optional YOLO ground-truth label (.txt) upload
# - Adds optional COCO JSON ground-truth upload
# - Computes per-image Precision(%), Recall(%), F1(%), using IoU matching
# - Adds these metrics as columns in the per-image Detections table and in the All Images summary table

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
from ultralytics import YOLO

# =========================
# Config
# =========================
ONNX_PATH = str(Path(__file__).parent / "models" / "best.onnx")
LOGO_PATH = str(Path(__file__).parent / "nova_logo.png")

st.set_page_config(page_title="ONNX Object Detection (Images)", layout="wide")

# =========================
# Main Title
# =========================
st.title("ONNX Object Detection (Images Only)")
st.caption("Upload image(s) → adjust confidence → run detection → save annotated outputs.")
st.markdown("---")

# =========================
# Sidebar with Logo and Controls
# =========================
# Logo in sidebar (smaller and non-clickable)
if Path(LOGO_PATH).exists():
    # Center the logo using columns
    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    with col2:
        st.image(LOGO_PATH, width=120)
else:
    st.sidebar.markdown("## NOVA")

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Settings")

# Image preprocessing
st.sidebar.subheader("Image Preprocessing")
brightness = st.sidebar.slider("Brightness adjustment (%)", -100, 100, 0, 10)

st.sidebar.subheader("Detection Parameters")
conf = st.sidebar.slider("Confidence threshold", 0.01, 1.00, 0.25, 0.01)
iou = st.sidebar.slider("IoU threshold (NMS)", 0.10, 0.95, 0.45, 0.01)
imgsz = st.sidebar.selectbox("Inference image size", [320, 416, 512, 640, 768, 1024], index=0)

st.sidebar.subheader("Visualization")
show_labels = st.sidebar.checkbox("Show labels", value=True)
show_conf = st.sidebar.checkbox("Show confidence", value=True)
line_width = st.sidebar.slider("Box line width", 1, 8, 2, 1)

st.sidebar.subheader("Metrics")
# IoU threshold for evaluation metrics (GT matching)
metric_iou = st.sidebar.slider("IoU threshold (metrics matching)", 0.10, 0.95, 0.50, 0.05)

# =========================
# Model load
# =========================
model_path = Path(ONNX_PATH)
if not model_path.exists():
    st.error(f"ONNX model not found: {model_path.resolve()}")
    st.stop()

@st.cache_resource
def load_model():
    return YOLO(ONNX_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load ONNX model at:\n{ONNX_PATH}\n\nError: {e}")
    st.stop()

# =========================
# Uploaders
# =========================
uploaded = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

# Ground truth format selection
gt_format = st.radio(
    "Ground truth format (optional)",
    ["None", "YOLO .txt files", "COCO JSON"],
    index=0
)

label_files = None
coco_json = None
coco_data = None

if gt_format == "YOLO .txt files":
    label_files = st.file_uploader(
        "Upload matching YOLO label .txt file(s)",
        type=["txt"],
        accept_multiple_files=True
    )
elif gt_format == "COCO JSON":
    coco_json = st.file_uploader(
        "Upload COCO annotations JSON file",
        type=["json"],
        accept_multiple_files=False
    )

# Resolution resize option
resize_option = st.radio(
    "Resize images (optional)",
    ["None", "1080p (1920x1080)", "720p (1280x720)"],
    index=0
)

# Map labels by stem/filename
labels_by_stem = {}

if label_files:
    for lf in label_files:
        labels_by_stem[Path(lf.name).stem] = lf

if coco_json:
    try:
        coco_data = json.load(coco_json)
        
        # Validate COCO structure
        if "images" not in coco_data:
            st.error("Invalid COCO JSON: missing 'images' field")
            coco_data = None
        elif "annotations" not in coco_data:
            st.error("Invalid COCO JSON: missing 'annotations' field")
            coco_data = None
        else:
            # Build image_id to filename mapping
            image_id_to_filename = {}
            for img in coco_data.get("images", []):
                image_id_to_filename[img["id"]] = {
                    "filename": img["file_name"],
                    "width": img.get("width", 0),
                    "height": img.get("height", 0)
                }
            
            # Group annotations by image_id (optional, for info display)
            annotations_count = len(coco_data.get("annotations", []))
            images_count = len(coco_data.get("images", []))
            
            st.success(f"✅ Loaded COCO JSON: {images_count} images, {annotations_count} annotations")
        
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse COCO JSON: Invalid JSON format - {e}")
        coco_data = None
    except Exception as e:
        st.error(f"Failed to load COCO JSON: {e}")
        coco_data = None

run = st.button("Run Detection", type="primary", disabled=(not uploaded))

# =========================
# Helpers
# =========================
def resize_image(pil_img: Image.Image, target_resolution: str) -> Image.Image:
    """
    Resize image to target resolution while maintaining aspect ratio.
    target_resolution: "1080p" or "720p" or "None"
    """
    if target_resolution == "None":
        return pil_img
    
    if target_resolution == "1080p (1920x1080)":
        target_width, target_height = 1920, 1080
    elif target_resolution == "720p (1280x720)":
        target_width, target_height = 1280, 720
    else:
        return pil_img
    
    # Get current dimensions
    current_width, current_height = pil_img.size
    
    # Calculate aspect ratios
    current_ratio = current_width / current_height
    target_ratio = target_width / target_height
    
    # Resize while maintaining aspect ratio
    if current_ratio > target_ratio:
        # Image is wider than target - fit to width
        new_width = target_width
        new_height = int(target_width / current_ratio)
    else:
        # Image is taller than target - fit to height
        new_height = target_height
        new_width = int(target_height * current_ratio)
    
    return pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def adjust_brightness(pil_img: Image.Image, brightness_pct: int) -> Image.Image:
    """
    Adjust image brightness.
    brightness_pct: -100 to 100 (percentage adjustment)
    """
    if brightness_pct == 0:
        return pil_img
    
    # Convert percentage to brightness factor
    # -100% = 0.0 (black), 0% = 1.0 (original), 100% = 2.0 (double brightness)
    factor = 1.0 + (brightness_pct / 100.0)
    enhancer = ImageEnhance.Brightness(pil_img)
    return enhancer.enhance(factor)

def annotate_pil(pil_img: Image.Image, result) -> Image.Image:
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    names = result.names if hasattr(result, "names") else {}

    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    boxes = result.boxes.xyxy.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c, p in zip(boxes, clss, confs):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # box
        for t in range(line_width):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(0, 255, 0))

        # label
        if show_labels:
            label = names.get(int(c), str(int(c)))
            if show_conf:
                label = f"{label} {float(p):.2f}"

            left, top, right, bottom = draw.textbbox((0, 0), label)
            tw, th = right - left, bottom - top
            pad = 4
            y0 = max(0, y1 - th - 2 * pad)

            draw.rectangle([x1, y0, x1 + tw + 2 * pad, y1], fill=(0, 255, 0))
            draw.text((x1 + pad, y0 + pad), label, fill=(0, 0, 0))

    return annotated

def box_iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def load_yolo_labels(file_like, img_w: int, img_h: int):
    """
    YOLO format lines:
      class x_center y_center width height   (all normalized 0..1)
    Returns list of dicts: [{"cls": int, "box": [x1,y1,x2,y2]}]
    """
    content = file_like.getvalue().decode("utf-8", errors="ignore").strip()
    if not content:
        return []

    gts = []
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h
        w = float(parts[3]) * img_w
        h = float(parts[4]) * img_h

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

        gts.append({"cls": cls_id, "box": [x1, y1, x2, y2]})
    return gts

def load_coco_annotations(filename: str, coco_data: dict):
    """
    Load COCO annotations for a specific image filename.
    COCO bbox format: [x, y, width, height] (absolute pixels)
    Returns list of dicts: [{"cls": int, "box": [x1,y1,x2,y2]}]
    """
    # Find image_id by filename
    image_id = None
    for img in coco_data.get("images", []):
        if img["file_name"] == filename or Path(img["file_name"]).name == filename:
            image_id = img["id"]
            break
    
    if image_id is None:
        return []
    
    # Get annotations for this image
    gts = []
    for ann in coco_data.get("annotations", []):
        if ann["image_id"] == image_id:
            bbox = ann.get("bbox", [])
            
            # Validate bbox format
            if not bbox or len(bbox) < 4:
                st.warning(f"Invalid bbox in annotation ID {ann.get('id', 'unknown')}: {bbox}")
                continue
            
            try:
                # COCO format: [x, y, width, height]
                x1 = float(bbox[0])
                y1 = float(bbox[1])
                w = float(bbox[2])
                h = float(bbox[3])
                x2 = x1 + w
                y2 = y1 + h
                
                # Skip invalid boxes (zero or negative area)
                if w <= 0 or h <= 0:
                    st.warning(f"Invalid bbox dimensions in annotation ID {ann.get('id', 'unknown')}: w={w}, h={h}")
                    continue
                
                # COCO category_id might need mapping to your model's class indices
                # For now, we'll use category_id - 1 (common COCO convention where class 0 is not used)
                cls_id = int(ann.get("category_id", 1)) - 1
                
                gts.append({"cls": cls_id, "box": [x1, y1, x2, y2]})
                
            except (ValueError, TypeError, IndexError) as e:
                st.warning(f"Error parsing bbox in annotation ID {ann.get('id', 'unknown')}: {e}")
                continue
    
    return gts

def compute_prf1(pred_boxes, pred_cls, gt_items, iou_thr: float):
    """
    Class-aware greedy matching:
    - A prediction is TP if it matches an unmatched GT of same class with IoU >= iou_thr.
    - Otherwise it is FP.
    - Unmatched GT are FN.
    Returns (tp, fp, fn, precision, recall, f1) where p/r/f1 are floats in [0,1].
    """
    if gt_items is None or len(gt_items) == 0:
        # No ground truth: all predictions are false positives
        fp = len(pred_boxes)
        return 0, fp, 0, 0.0, 0.0, 0.0

    matched_gt = [False] * len(gt_items)
    tp = 0
    fp = 0

    for pb, pc in zip(pred_boxes, pred_cls):
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_items):
            if matched_gt[j]:
                continue
            if int(gt["cls"]) != int(pc):
                continue
            iou_val = box_iou_xyxy(pb, gt["box"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j

        if best_j >= 0 and best_iou >= iou_thr:
            matched_gt[best_j] = True
            tp += 1
        else:
            fp += 1

    fn = int(len(gt_items) - sum(matched_gt))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1

# =========================
# Run
# =========================
if run:
    all_conf_rows = []

    for i, f in enumerate(uploaded):
        st.markdown("---")
        st.subheader(f"Image {i+1}: {f.name}")

        # Load image
        pil_original = Image.open(f).convert("RGB")
        original_size = pil_original.size
        
        # Apply resize if selected
        pil_resized = resize_image(pil_original, resize_option)
        resized_size = pil_resized.size
        
        # Apply brightness adjustment
        pil = adjust_brightness(pil_resized, brightness)
        
        img_rgb = np.array(pil)
        img_w, img_h = pil.size

        # Show size info if resized
        if resize_option != "None":
            st.info(f"📐 Resized from {original_size[0]}x{original_size[1]} to {resized_size[0]}x{resized_size[1]}")

        results = model.predict(
            source=img_rgb,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )
        r = results[0]

        # Predictions (xyxy + cls + conf)
        if r.boxes is not None and len(r.boxes) > 0:
            pred_boxes = r.boxes.xyxy.cpu().numpy().astype(float).tolist()
            pred_cls = r.boxes.cls.cpu().numpy().astype(int).tolist()
            pred_confs = r.boxes.conf.cpu().numpy().astype(float).tolist()
        else:
            pred_boxes, pred_cls, pred_confs = [], [], []

        # Load ground truth labels based on format
        gt_items = None
        
        if gt_format == "YOLO .txt files" and labels_by_stem:
            stem = Path(f.name).stem
            if stem in labels_by_stem:
                gt_items = load_yolo_labels(labels_by_stem[stem], img_w, img_h)
                st.info(f"✅ Loaded {len(gt_items)} YOLO ground truth boxes")
        
        elif gt_format == "COCO JSON" and coco_json:
            # Try to match by exact filename or just the filename part
            if coco_data is not None:
                gt_items = load_coco_annotations(f.name, coco_data)
                if len(gt_items) == 0:
                    # Try with just the basename
                    gt_items = load_coco_annotations(Path(f.name).name, coco_data)
                
                if len(gt_items) > 0:
                    st.info(f"✅ Loaded {len(gt_items)} COCO ground truth boxes")
                else:
                    st.warning(f"⚠️ No COCO annotations found for {f.name}")
            else:
                st.error("COCO data not loaded properly")

        # Compute P/R/F1 (only if GT exists; else show "-")
        if gt_items is not None and len(gt_items) > 0:
            tp, fp, fn, prec, rec, f1 = compute_prf1(pred_boxes, pred_cls, gt_items, metric_iou)
            st.write(f"📊 Metrics - TP: {tp}, FP: {fp}, FN: {fn} | GT boxes: {len(gt_items)}, Predictions: {len(pred_boxes)}")
            prec_pct = prec * 100.0
            rec_pct = rec * 100.0
            f1_pct = f1 * 100.0
        else:
            tp = fp = fn = 0
            prec_pct = rec_pct = f1_pct = None

        # Per-image confidence summary (min/mean/max)
        if len(pred_confs) > 0:
            row = {
                "image": f.name,
                "num_detections": int(len(pred_confs)),
                "conf_min": float(np.min(pred_confs)),
                "conf_mean": float(np.mean(pred_confs)),
                "conf_max": float(np.max(pred_confs)),
                "precision(%)": float(prec_pct) if prec_pct is not None else np.nan,
                "recall(%)": float(rec_pct) if rec_pct is not None else np.nan,
                "f1_score(%)": float(f1_pct) if f1_pct is not None else np.nan,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
            }
        else:
            row = {
                "image": f.name,
                "num_detections": 0,
                "conf_min": np.nan,
                "conf_mean": np.nan,
                "conf_max": np.nan,
                "precision(%)": float(prec_pct) if prec_pct is not None else np.nan,
                "recall(%)": float(rec_pct) if rec_pct is not None else np.nan,
                "f1_score(%)": float(f1_pct) if f1_pct is not None else np.nan,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
            }

        all_conf_rows.append(row)

        # Show per-image summary table
        conf_summary = pd.DataFrame([row]).copy()
        for col in ["conf_min", "conf_mean", "conf_max", "precision(%)", "recall(%)", "f1_score(%)"]:
            conf_summary[col] = conf_summary[col].map(lambda x: "-" if pd.isna(x) else f"{float(x):.3f}")
        st.write("Confidence + Metrics summary (this image):")
        st.table(conf_summary)

        annotated_pil = annotate_pil(pil, r)

        c1, c2 = st.columns(2)
        with c1:
            # Build caption with resize and brightness info
            caption_parts = ["Original"]
            if resize_option != "None":
                caption_parts.append(f"Resized to {resized_size[0]}x{resized_size[1]}")
            if brightness != 0:
                caption_parts.append(f"Brightness: {brightness:+d}%")
            caption_text = " | ".join(caption_parts)
            
            st.image(pil, caption=caption_text, use_container_width=True)
        with c2:
            st.image(annotated_pil, caption=f"Detected (conf ≥ {conf:.2f})", use_container_width=True)

        # Download annotated image
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        st.download_button(
            label=f"Download annotated: {Path(f.name).stem}_detected.png",
            data=buf.getvalue(),
            file_name=f"{Path(f.name).stem}_detected.png",
            mime="image/png",
            key=f"dl_{Path(f.name).stem}_{i}_{conf:.2f}_{iou:.2f}"
        )

        # Per-image detections table with P/R/F1 columns
        if len(pred_boxes) > 0:
            st.write("Detections:")
            names = r.names

            det_rows = []
            for (x1, y1, x2, y2), c, p in zip(pred_boxes, pred_cls, pred_confs):
                det_rows.append({
                    "class_id": int(c),
                    "class_name": names.get(int(c), str(int(c))),
                    "confidence": float(p),

                    # Per-image metrics repeated per row (needs GT)
                    "precision(%)": "-" if prec_pct is None else round(float(prec_pct), 3),
                    "recall(%)": "-" if rec_pct is None else round(float(rec_pct), 3),
                    "f1_score(%)": "-" if f1_pct is None else round(float(f1_pct), 3),

                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)
                })

            st.dataframe(pd.DataFrame(det_rows), use_container_width=True)

            if gt_items is None or len(gt_items) == 0:
                st.info("Metrics (Precision/Recall/F1) show '-' because no matching ground truth was found for this image.")
        else:
            st.info("No detections for this image at the current confidence threshold.")

    # All images summary
    st.markdown("## Confidence Summary (All Images)")
    all_conf_df = pd.DataFrame(all_conf_rows)

    display_df = all_conf_df.copy()
    for col in ["conf_min", "conf_mean", "conf_max", "precision(%)", "recall(%)", "f1_score(%)"]:
        display_df[col] = display_df[col].map(lambda x: "-" if pd.isna(x) else f"{float(x):.3f}")

    st.dataframe(display_df, use_container_width=True)

    csv_bytes = all_conf_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download confidence summary CSV (all images)",
        data=csv_bytes,
        file_name="confidence_summary_all_images.csv",
        mime="text/csv",
        key=f"dl_conf_summary_{len(all_conf_df)}_{conf:.2f}_{iou:.2f}"
    )

else:
    st.info("Upload image(s) and click **Run Detection** to generate results.")

st.info(f"Loaded ONNX model: {ONNX_PATH}")