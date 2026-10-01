# PCOS Ultrasound Region Detector

A YOLOv8-based object detector that identifies **infected** vs. **not-infected** ovarian regions in ultrasound images — a screening-aid prototype relevant to PCOS (Polycystic Ovary Syndrome) diagnostics.

**Live demo:** https://pcos-region-detector-4j5hh47rp6jcyysjfqynhf.streamlit.app/
**Model:** YOLOv8n, fine-tuned on ultrasound imagery

---

## Problem

PCOS affects an estimated 1 in 10 women of reproductive age and is commonly assessed via ultrasound evaluation of ovarian morphology. Manual review of ultrasound frames is time-consuming and subjective. This project explores whether a lightweight object detection model can flag infected (PCOS-indicative) regions in an ultrasound frame automatically, as a potential screening aid.

## Dataset

- Source: PCOS Detection (Roboflow Universe), 4,530 annotated ultrasound images
- Classes: `infected`, `notinfected`
- Split: 3,954 train / 384 validation / ~190 test
- **Limitation:** community-labeled dataset, not clinically validated or sourced from a peer-reviewed clinical study. Treat results as a proof-of-concept, not diagnostic ground truth.

## Model & Training

- Architecture: YOLOv8n (nano), transfer-learned from COCO pretrained weights
- Image size: 416x416
- Epochs: 30
- Hardware: trained CPU-only (12-core laptop, ~3.9 hours total) — included here deliberately, since GPU access was not available; demonstrates the project is reproducible without specialized hardware
- Framework: Ultralytics YOLOv8

## Results

| Metric | Score |
|---|---|
| mAP50 | 0.993 |
| mAP50-95 | 0.555 |
| Precision | 0.991 |
| Recall | 0.994 |

Per-class mAP50: infected 0.993, notinfected 0.993 — balanced performance across both classes, no class is systematically under-detected.

## Project Note: Scope Pivot

This project was originally scoped as an **antral follicle counting** tool (segmenting and counting individual follicles from ultrasound, relevant to AFC-based fertility/PCOS workups). During dataset selection, the most viable publicly available annotated dataset in the 48-hour build window turned out to be labeled for **infected/notinfected region classification**, not individual follicle instances.

Rather than force-fit mismatched labels to the original framing, the project was reframed around what the data actually supports: region-level PCOS-pattern detection. This is a judgment call that comes up often in applied ML/analytics work — validate what a dataset actually contains before committing a deliverable's framing to it.

## Limitations

- Binary region classification, not follicle-level instance counting or clinical AFC (antral follicle count)
- Community-labeled dataset — no clinical validation or inter-rater agreement reported
- Single 2D ultrasound frames — no 3D volume or temporal/video context
- Not intended for diagnostic use; a research/portfolio prototype only

## Next Steps (v2)

- True follicle-level instance segmentation (U-Net or YOLOv8-seg) on a clinically validated dataset, e.g. USOVA3D (42k annotated images, requires registered access)
- Pixel-to-mm calibration for follicle size estimation and dominant-follicle (>=18mm) flagging
- Multi-frame / cine-loop aggregation instead of single-frame inference

## Usage

\`\`\`bash
pip install ultralytics gradio
python3 app.py
\`\`\`
Upload an ultrasound image via the Gradio interface to get annotated detections and a region count summary.

## Tech Stack

YOLOv8 (Ultralytics) · Python · Gradio · Roboflow (data pipeline)
