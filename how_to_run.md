# How to Run — PillOCR Streamlit App

## Prerequisites

- **Python 3.10+** (via Conda)
- **macOS** with Apple Silicon (MPS) or NVIDIA GPU (CUDA)

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```


## 3. Prepare Models

Place model weights under `app/models/`:

```
app/models/
├── YOLO/
│   └── <model_name>/
│       └── best.pt          # YOLO detection weights (AABB or OBB)
└── trocr_finetuned_best/    # Fine-tuned TrOCR model directory
    ├── config.json
    ├── preprocessor_config.json
    └── pytorch_model.bin (or model.safetensors)
```

- **YOLO models** are auto-scanned from `app/models/YOLO/`. Each subfolder containing a `best.pt` becomes a selectable option in the UI.
- **TrOCR model** — if `app/models/trocr_finetuned_best/` exists locally, it's used. Otherwise, the model is **auto-downloaded from Kaggle** (`foxkawaii/vaipe-trocr/transformers/default`) on first run.

## 4. Run the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

## Usage

### Mode 1 — Full Image (Detect Pills)

1. Select a YOLO model from the sidebar (OBB models are auto-detected).
2. Adjust **Confidence** and **IoU** thresholds.
3. Upload an image → click **Run OCR Pipeline**.
4. The app runs YOLO to detect pill/text_zone regions, then TrOCR to read text.

### Mode 2 — Pre-cropped Pill (OCR Only)

1. Switch to **Pre-cropped Pill (OCR Only)** in the sidebar.
2. Upload a cropped pill image → click **Run OCR Pipeline**.
3. TrOCR runs directly without YOLO detection.

### Crop Transform Options

| Option | Description |
|---|---|
| **Auto-orient** | Tries 24 rotation+flip combos, picks the best by majority vote |
| **Manual rotation** | 0°, 90°, 180°, 270° |
| **Flip** | None, Horizontal, Vertical |

## Troubleshooting

| Issue | Fix |
|---|---|
| `No YOLO models found` | Ensure `app/models/YOLO/` contains at least one subfolder with `best.pt` |
| Slow first run | Model loading is cached after the first run via `@st.cache_resource` |
| MPS errors on macOS | Update PyTorch: `pip install --upgrade torch` |
| CUDA out of memory | Reduce image size or use CPU: set `CUDA_VISIBLE_DEVICES=""` |
