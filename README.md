# NOVA Low Light Vehicle Speed Detection Research Tool

A Streamlit web application designed to streamline experiments for low-light vehicle speed detection research. This tool enables researchers to systematically evaluate how various environmental and technical parameters affect object detection model accuracy.

![NOVA Logo](nova_logo.png)

## Overview

This research tool provides a comprehensive platform for testing and validating vehicle detection models under different conditions. It supports ONNX models and provides detailed metrics including precision, recall, and F1 scores with ground truth comparison capabilities.

## Key Features

### Experimental Simulations

The tool enables systematic testing of multiple factors affecting model accuracy:

#### 1. **Camera Resolution Effects**
- Resize images to standard resolutions (1080p, 720p)
- Evaluate how resolution changes impact detection accuracy
- Maintain aspect ratio during resizing for realistic testing

#### 2. **Mounting Height Simulation**
- Adjust image perspective to simulate different camera mounting heights
- Test model performance across various installation scenarios

#### 3. **Brightness/Lighting Conditions**
- Adjust brightness from -100% to +100% in 10% increments
- Simulate low-light, normal, and over-exposed conditions
- Critical for testing low-light vehicle detection scenarios

#### 4. **Confidence Threshold Optimization**
- Fine-tune detection confidence thresholds (0.01 to 1.00)
- Balance between false positives and false negatives
- Adjustable IoU threshold for Non-Maximum Suppression (NMS)

#### 5. **Ground Truth Comparison**
- **Manual vs AI Detection**: Compare human-annotated ground truth with AI predictions
- **Multiple Format Support**:
  - YOLO .txt format (normalized coordinates)
  - COCO JSON format (absolute pixel coordinates)
- **Detailed Metrics**:
  - True Positives (TP)
  - False Positives (FP)
  - False Negatives (FN)
  - Precision (%)
  - Recall (%)
  - F1 Score (%)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Packages

Create a `requirements.txt` file:

```txt
streamlit
numpy
pandas
pillow
opencv-python-headless==4.10.0.84
ultralytics==8.2.90
onnxruntime
```

### System Dependencies (for Streamlit Cloud)

Create a `packages.txt` file:

```txt
libgl1-mesa-glx
libglib2.0-0
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
nova-vehicle-detection/
├── app_onnx.py              # Main Streamlit application
├── nova_logo.png            # NOVA project logo
├── models/
│   └── best.onnx           # ONNX model file
├── requirements.txt         # Python dependencies
├── packages.txt            # System dependencies
└── README.md               # This file
```

## Usage

### Starting the Application

```bash
streamlit run app_onnx.py
```

Or with auto-reload on save:

```bash
streamlit run app_onnx.py --server.runOnSave true
```

### Basic Workflow

1. **Upload Images**: Select one or multiple test images (JPG, JPEG, PNG, WEBP)

2. **Configure Settings** (Sidebar):
   - **Image Preprocessing**: Adjust brightness
   - **Detection Parameters**: Set confidence and IoU thresholds
   - **Visualization**: Toggle labels, confidence scores, box width
   - **Metrics**: Configure IoU threshold for ground truth matching

3. **Optional: Upload Ground Truth**:
   - Choose format: YOLO .txt or COCO JSON
   - Upload corresponding annotation files

4. **Optional: Resize Images**:
   - Select target resolution (1080p or 720p)

5. **Run Detection**: Click "Run Detection" button

6. **Analyze Results**:
   - View side-by-side original and annotated images
   - Review per-image metrics and confidence statistics
   - Download annotated images
   - Export comprehensive CSV reports

### Ground Truth Formats

#### YOLO Format (.txt)
Each line represents one object:
```
class_id x_center y_center width height
```
All values normalized to [0, 1]

Example:
```
0 0.5 0.5 0.3 0.4
```

#### COCO Format (JSON)
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height]
    }
  ]
}
```

## Output and Metrics

### Per-Image Metrics

- **Detection Count**: Number of objects detected
- **Confidence Statistics**: Min, Mean, Max confidence scores
- **Precision**: TP / (TP + FP) × 100%
- **Recall**: TP / (TP + FN) × 100%
- **F1 Score**: Harmonic mean of precision and recall

### Export Options

1. **Annotated Images**: Download individual processed images with bounding boxes
2. **CSV Report**: Comprehensive summary of all images with metrics
3. **Detection Tables**: Detailed per-detection information with coordinates

## Research Applications

### Experimental Design

This tool supports systematic research on:

1. **Optimal Camera Placement**
   - Test different mounting heights
   - Analyze detection accuracy vs. installation cost

2. **Lighting Condition Studies**
   - Day vs. night performance
   - Impact of streetlight positioning
   - Low-light enhancement effectiveness

3. **Hardware Requirements**
   - Minimum resolution for acceptable accuracy
   - Cost-performance trade-offs

4. **Model Optimization**
   - Confidence threshold calibration
   - IoU threshold tuning
   - Precision-recall balance

### Data Collection and Analysis

- Batch process multiple images for statistical significance
- Export results for further analysis in Excel, Python, R
- Track performance across different experimental conditions
- Generate publication-ready metrics

## Configuration Options

### Sidebar Settings

| Setting | Range/Options | Default | Purpose |
|---------|--------------|---------|---------|
| Brightness | -100% to +100% | 0% | Simulate lighting conditions |
| Confidence Threshold | 0.01 to 1.00 | 0.25 | Detection sensitivity |
| IoU (NMS) | 0.10 to 0.95 | 0.45 | Overlap threshold |
| Inference Size | 320-1024 pixels | 320 | Model input size |
| Box Line Width | 1-8 pixels | 2 | Visualization |
| Metrics IoU | 0.10 to 0.95 | 0.50 | Ground truth matching |

## Troubleshooting

### Common Issues

**Model not found**
- Ensure `best.onnx` is in the `models/` directory
- Check file permissions

**Import errors (OpenCV)**
- Install system dependencies from `packages.txt`
- Use `opencv-python-headless` for server environments

**Memory issues with large images**
- Use the resize option to reduce image dimensions
- Process images in smaller batches

**Ground truth not matching**
- Verify filename matches (without extension)
- Check annotation format (YOLO vs COCO)
- Ensure coordinates are correctly normalized/absolute

## Performance Tips

1. **Batch Processing**: Upload multiple images at once for efficiency
2. **Resolution**: Use 720p for faster processing during initial testing
3. **Cache**: Model loads once and is cached for the session
4. **Ground Truth**: Pre-organize annotations by filename for easier batch testing

## Contributing

This tool is developed for the NOVA Low Light Vehicle Speed Detection Research project. For questions or contributions, please contact the research team.

## License

[Specify your license here]

## Citation

If you use this tool in your research, please cite:

```
[Your citation format here]
```

## Contact

**Author**: Clarence Jay Fetalino  
**Project**: NOVA Low Light Vehicle Speed Detection Research

---

**Version**: 1.0  
**Last Updated**: 2026
