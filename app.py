import streamlit as st
import tempfile
import os
from pathlib import Path
import time
import torch
import kagglehub
from ultralytics import YOLO
from inference_utils import (
    predict_pill_ocr, load_trocr_model, predict_cropped_pill_ocr, is_obb_model,
    brute_force_best_orientation, load_known_texts
)

# Constants
APP_DIR = Path(__file__).resolve().parent
YOLO_MODELS_DIR = str(APP_DIR / "models" / "YOLO")
TROCR_LOCAL_PATH = str(APP_DIR / "models" / "trocr_finetuned_best")
KAGGLE_MODEL_SLUG = "foxkawaii/vaipe-trocr/transformers/default"


def resolve_trocr_path():
    """Use local model if available, otherwise download from Kaggle."""
    if os.path.isdir(TROCR_LOCAL_PATH):
        return TROCR_LOCAL_PATH
    return kagglehub.model_download(KAGGLE_MODEL_SLUG)

st.set_page_config(layout="wide")


def scan_yolo_models(models_dir):
    """Scan the YOLO models directory for available model weights.
    
    Looks for:
    - best.pt directly in models_dir  (legacy layout)
    - best.pt inside subdirectories   (versioned layout)
    
    Returns dict: { display_name: full_path }
    """
    models = {}
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return models
    
    # Check for best.pt directly in the dir
    direct = models_path / "best.pt"
    if direct.exists():
        models["default (best.pt)"] = str(direct)
    
    # Check subdirectories
    for subdir in sorted(models_path.iterdir()):
        if subdir.is_dir():
            weight = subdir / "best.pt"
            if weight.exists():
                models[subdir.name] = str(weight)
            # Also check for any .pt files
            for pt_file in sorted(subdir.glob("*.pt")):
                key = f"{subdir.name}/{pt_file.name}"
                if key not in models.values():
                    models[key] = str(pt_file)
    
    # Check for any .pt files directly in the dir (other than best.pt)
    for pt_file in sorted(models_path.glob("*.pt")):
        if pt_file.name != "best.pt":
            models[pt_file.stem] = str(pt_file)
    
    return models


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@st.cache_resource
def load_yolo_model(model_path):
    device = get_device()
    model = YOLO(model_path)
    model.to(device)
    return model


@st.cache_resource
def get_trocr_model():
    device = get_device()
    trocr_path = resolve_trocr_path()
    processor, model = load_trocr_model(trocr_path, device)
    return processor, model, device


# ─── Sidebar: Model Selection & Settings ───────────────────────────

with st.sidebar:
    st.header("🔬 Model Selection")
    
    available_models = scan_yolo_models(YOLO_MODELS_DIR)
    
    if not available_models:
        st.error(f"No YOLO models found in `{YOLO_MODELS_DIR}`")
        st.stop()
    
    model_names = list(available_models.keys())
    selected_model_name = st.selectbox(
        "YOLO Model",
        model_names,
        index=0,
        help="Select a YOLO model. OBB models are auto-detected."
    )
    selected_model_path = available_models[selected_model_name]
    
    st.divider()
    st.header("⚙️ Detection Settings")
    input_type = st.radio("Input Type", ["Full Image (Detect Pills)", "Pre-cropped Pill (OCR Only)"])
    if input_type == "Full Image (Detect Pills)":
        conf = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        iou = st.slider("IoU threshold (NMS)", min_value=0.0, max_value=1.0, value=0.45, step=0.05)

    st.divider()
    st.header("🔄 Crop Transform")
    auto_orient = st.checkbox(
        "🔍 Auto-orient (brute-force)",
        value=False,
        help="Try all 12 rotation+flip combos and pick the one whose OCR matches a known pill text. Slower but automatic."
    )
    if not auto_orient:
        st.caption("Adjust if text zone crops appear rotated")
        crop_rotation = st.select_slider(
            "Rotation",
            options=[0, 90, 180, 270],
            value=0,
            format_func=lambda x: f"{x}°"
        )
        crop_flip = st.selectbox(
            "Flip",
            ["None", "Horizontal", "Vertical"],
            index=0
        )
    else:
        crop_rotation = 0
        crop_flip = "None"
        st.caption("Will try all 4 rotations × 3 flips × 2 methods = **24 OCR attempts** per text zone")


# ─── Load Models ───────────────────────────────────────────────────

st.title("Pill OCR Detection App")

with st.spinner("Loading models..."):
    yolo_model = load_yolo_model(selected_model_path)
    trocr_processor, trocr_model, trocr_device = get_trocr_model()

# Show model info
obb_mode = is_obb_model(yolo_model)
model_type_badge = "🔄 OBB (Oriented)" if obb_mode else "⬜ AABB (Axis-Aligned)"
class_list = ", ".join(yolo_model.names.values())
st.caption(f"**Model:** `{selected_model_name}` · **Type:** {model_type_badge} · **Classes:** {class_list}")


# ─── Upload & Run ──────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if st.button("Run OCR Pipeline") and uploaded_file is not None:
    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name
    
    st.info("Processing Image...")
    try:
        if input_type == "Full Image (Detect Pills)":
            final_image, output_path, pill_results = predict_pill_ocr(
                yolo_model=yolo_model,
                trocr_processor=trocr_processor,
                trocr_model=trocr_model,
                source_path=input_path,
                trocr_device=trocr_device,
                conf=conf,
                iou_threshold=iou,
                crop_rotation=crop_rotation,
                crop_flip=crop_flip,
                auto_orient=auto_orient
            )
            
            total_time = time.time() - start_time
            st.success(f"Processing completed in {total_time:.2f}s!")
            
            # Display main image
            st.subheader("Original Image with Detections")
            st.image(output_path, caption="Pill OCR Result", use_container_width=True)
        else:
            pill_results = predict_cropped_pill_ocr(
                trocr_processor=trocr_processor,
                trocr_model=trocr_model,
                source_path=input_path,
                trocr_device=trocr_device,
                auto_orient=auto_orient
            )
            total_time = time.time() - start_time
            st.success(f"Processing completed in {total_time:.2f}s!")
        
        # Display each text zone OCR result
        if pill_results:
            st.subheader("Text Zone OCR Results")
            for i, res in enumerate(pill_results):
                st.markdown(f"**Text Zone #{i+1}**")

                # Show auto-orient info if available
                bf = res.get('auto_orient')
                if bf is not None:
                    if bf.get("is_numeric"):
                        match_icon = "🔢"
                    else:
                        match_icon = "📝"
                    votes = bf.get("vote_count", "?")
                    total = bf.get("total_attempts", "?")
                    st.markdown(
                        f"{match_icon} **Best text:** `{bf['best_text']}` "
                        f"(**{votes}/{total}** votes) "
                        f"— rotation **{bf['rotation']}°**, flip **{bf['flip']}**, "
                        f"method **{bf['best_method']}**"
                    )

                col0, col1, col2 = st.columns(3)
                
                with col0:
                    st.image(res['original_crop'], caption="Best Crop" if bf else "Original Crop", width=200)
                
                with col1:
                    st.image(res['processed_1'], caption="Method 1: Adaptive + Inverted Closing", width=200)
                    st.write(f"**Recognized Text:** `{res['text_1']}`")
                
                with col2:
                    st.image(res['processed_2'], caption="Method 2: CLAHE + Adaptive", width=200)
                    st.write(f"**Recognized Text:** `{res['text_2']}`")

                # Expandable table of all 24 attempts
                if bf is not None and bf.get('all_results'):
                    with st.expander(f"🔬 All {len(bf['all_results'])} orientation attempts"):
                        import pandas as pd
                        df = pd.DataFrame(
                            bf['all_results'],
                            columns=['Rotation', 'Flip', 'Method', 'Text']
                        )
                        # Add vote count column
                        vote_map = df['Text'].str.strip().str.upper().map(
                            df['Text'].str.strip().str.upper().value_counts()
                        )
                        df['Votes'] = vote_map
                        st.dataframe(df, use_container_width=True, hide_index=True)

                st.divider()
        else:
            st.warning("No text zones detected in the image.")
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)
elif uploaded_file is None:
    st.info("Please upload an image to start.")