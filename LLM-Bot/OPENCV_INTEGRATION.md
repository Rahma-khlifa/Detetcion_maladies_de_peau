# OpenCV Integration - LLM-Bot

## Overview

OpenCV has been successfully integrated into LLM-Bot to provide **automated image analysis** of dermatological lesions. This creates a comprehensive analysis system combining:

1. **OpenCV Feature Extraction** - Automated ABCDE melanoma criteria analysis
2. **RAG (Retrieval-Augmented Generation)** - Medical literature search based on visual features
3. **LLM Interpretation** - Clinical interpretation using Flan-T5-XL

## What Was Added

### 1. OpenCV Service (`src/services/opencv_service.py`)
Copied from VLM-Bot, provides:

- **Lesion Segmentation**: K-means clustering to isolate lesion from background
- **Shape Analysis**: Asymmetry, circularity, diameter
- **Color Analysis**: Dominant colors with dermatological classification
- **Texture Analysis**: Pigmentation patterns, surface characteristics
- **Border Assessment**: Regularity and definition
- **ABCDE Risk Factors**: Automated melanoma risk screening

### 2. Configuration (`config.yaml`)
Added OpenCV section:
```yaml
opencv:
  max_image_dim: 1024
  segmentation_clusters: 3
  min_lesion_size: 500
  n_colors: 4
  pixels_per_mm: 10
```

### 3. Dependencies (`requirements.txt`)
Added vision processing libraries:
- `Pillow>=10.1.0` - Image loading
- `opencv-python-headless>=4.8.1.78` - Computer vision
- `imutils>=0.5.4` - Image utilities
- `scikit-image>=0.22.0` - Morphological operations
- `webcolors>=1.13` - Color naming

### 4. Application Interface (`src/app.py`)

#### New Tab: 🔬 Analyse d'Image
3-phase analysis workflow:

**Phase 1: OpenCV Analysis**
- Upload dermatological lesion image
- Automatic extraction of visual features
- Generates detailed ABCDE report

**Phase 2: RAG Search**
- Constructs intelligent search query from OpenCV findings
- Example: "asymmetric lesion irregular border multiple colors"
- Retrieves relevant medical literature

**Phase 3: LLM Interpretation**
- Combines OpenCV analysis + RAG sources
- Generates clinical interpretation
- Provides differential diagnosis and recommendations

## How It Works

### Image Analysis Workflow

```
User Uploads Image
        ↓
OpenCV Service Analyzes Image
  ├─ Segment lesion (K-means)
  ├─ Extract shape features (asymmetry, circularity)
  ├─ Analyze colors (dominant colors, distribution)
  ├─ Assess texture (pigmentation patterns)
  ├─ Evaluate borders (regularity, definition)
  └─ Calculate ABCDE risk factors
        ↓
RAG Service Searches Literature
  ├─ Build query from OpenCV features
  │   (e.g., "asymmetric irregular border black pigmentation")
  ├─ Search FAISS index
  └─ Retrieve top-k relevant abstracts
        ↓
LLM Service Generates Interpretation
  ├─ Input: OpenCV analysis + RAG sources
  ├─ Prompt: Request clinical interpretation
  └─ Output: Differential diagnosis, recommendations
        ↓
Save Complete Report
```

### Example OpenCV Output

```
DERMATOLOGICAL LESION ANALYSIS:

MORPHOLOGY:
- Size: Approximately 7.2mm diameter, 40.8mm² area
- Shape: markedly asymmetric with irregular borders
- Border definition: poorly-defined
- Overall circularity: 0.245 (1.0 = perfect circle)

COLOR ANALYSIS:
- Number of distinct color zones: 4
- Color composition: dark brown 45.2% (central); 
  medium brown 28.1% (peripheral); 
  black/very dark brown 18.3% (mixed); 
  pink 8.4% (peripheral)
- Color pattern: Variegated (multiple distinct colors)

ABCDE MELANOMA RISK FACTORS:
⚠️  A: Significant asymmetry detected
⚠️  B: Highly irregular border
⚠️  C: Multiple colors present (≥4 distinct tones)
⚠️  C: Black pigmentation present
⚠️  D: Diameter > 6mm (7.2mm)
```

### Example RAG Query Construction

Based on OpenCV findings:
- Asymmetry > 30 → Add "asymmetric lesion"
- Irregular border → Add "irregular border"
- ≥3 colors → Add "multiple colors variegated"
- Black pigmentation → Add "black pigmentation"
- Diameter > 6mm → Add "large lesion melanoma"

Final query: `"asymmetric lesion irregular border multiple colors black pigmentation large lesion melanoma"`

## Usage

### Basic Image Analysis

1. Navigate to **🔬 Analyse d'Image** tab
2. Upload a dermatological lesion image
3. Enable RAG (recommended)
4. Click **🔬 Analyser l'Image**

### View Results

- **🔬 Analyse OpenCV**: Detailed visual feature extraction
- **📚 Sources Médicales**: Retrieved medical literature
- **🏥 Interprétation Clinique**: LLM-generated clinical interpretation

### Advanced Parameters

- **Nombre de sources** (1-10): Number of medical abstracts to retrieve
- **Tokens de génération** (128-1024): LLM response length
- **Température** (0.1-1.0): LLM creativity (higher = more diverse)

## Installation

### Install New Dependencies

```bash
# Activate conda environment
conda activate rag

# Navigate to LLM-Bot
cd E:\Chatbots\LLM-Bot

# Install OpenCV dependencies
pip install opencv-python-headless>=4.8.1.78 imutils>=0.5.4 scikit-image>=0.22.0 webcolors>=1.13
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Comparison: LLM-Bot vs VLM-Bot

| Feature | LLM-Bot | VLM-Bot |
|---------|---------|---------|
| **Image Analysis** | OpenCV (classical CV) | Phi-3-Vision (deep learning) |
| **Text Analysis** | Flan-T5-XL (3B params) | N/A |
| **RAG** | ✅ FAISS | ✅ FAISS |
| **VRAM Usage** | ~2-3 GB | ~2.3 GB |
| **Strengths** | Quantitative ABCDE metrics, fast | Natural language understanding of images |
| **Use Case** | Objective measurements + text Q&A | Subjective visual interpretation |

## Output Files

Each analysis saves a comprehensive report:
- **Filename**: `image_analysis_YYYYMMDD_HHMMSS.txt`
- **Content**: 
  - OpenCV feature extraction
  - Retrieved medical literature
  - LLM clinical interpretation
  - Disclaimer

## Advantages of OpenCV Integration

### 1. **Quantitative Analysis**
- Precise measurements (diameter, area, asymmetry score)
- Objective color quantification
- Reproducible metrics

### 2. **ABCDE Screening**
- Automated melanoma risk factor detection
- Immediate red flags
- Clinical decision support

### 3. **RAG Enhancement**
- Intelligent query construction from visual features
- More relevant literature retrieval
- Evidence-based interpretation

### 4. **No Additional VRAM**
- OpenCV runs on CPU
- No GPU memory overhead
- Complements LLM efficiently

### 5. **Interpretable Features**
- Human-readable measurements
- Clinically relevant terminology
- Explainable AI

## Limitations

1. **Segmentation Accuracy**: K-means may fail on complex backgrounds
2. **Calibration**: Pixel-to-mm conversion assumes default 10 px/mm
3. **Lighting Dependency**: Color analysis sensitive to illumination
4. **No Deep Features**: Classical CV vs. learned representations

## Next Steps

### Recommended Improvements

1. **Calibration Tool**: Add UI for px/mm calibration
2. **Segmentation Fallback**: Manual ROI selection if auto-segmentation fails
3. **Comparative Analysis**: Side-by-side comparison with previous images
4. **Export Visualization**: Save annotated images with measurements

### Testing

Test with sample dermatological images:
```bash
cd E:\Chatbots\LLM-Bot
conda run -n rag python src\app.py
```

Then:
1. Open http://localhost:7861
2. Navigate to 🔬 Analyse d'Image
3. Upload test image
4. Review OpenCV extraction, RAG sources, and LLM interpretation

## Troubleshooting

### OpenCV Import Error
```bash
pip install opencv-python-headless --upgrade
```

### Scikit-image Error
```bash
pip install scikit-image --upgrade
```

### Webcolors Error
```bash
pip install webcolors
```

### Memory Issues
OpenCV runs on CPU - if RAM is low, reduce `max_image_dim` in config.yaml

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        LLM-Bot                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Tab 1: 🔬 Image Analysis (NEW)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   OpenCV     │→ │     RAG      │→ │   Flan-T5    │     │
│  │  Features    │  │   Search     │  │     XL       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  Tab 2: 🩺 Symptom Analysis                                │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  Text Input  │→ │     RAG      │→ │   Flan-T5    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  Tab 3: ❓ General Q&A                                     │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │   Question   │→ │     RAG      │→ │   Flan-T5    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Credits

- **OpenCV Service**: Originally developed for VLM-Bot
- **Integration**: Adapted for LLM-Bot text-based workflow
- **Dataset**: TimSchopf/medical_abstracts (Hugging Face)

---

**Status**: ✅ Integration Complete  
**Date**: January 3, 2026  
**Version**: LLM-Bot v1.1 (with OpenCV)
