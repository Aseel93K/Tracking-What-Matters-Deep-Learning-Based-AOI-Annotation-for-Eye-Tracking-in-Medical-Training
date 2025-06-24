# 🧠 Mask R-CNN Training with MMDetection

This project uses [MMDetection](https://github.com/open-mmlab/mmdetection) to train a Mask R-CNN model for instance segmentation on a custom dataset.

---

## 📁 Project Structure
```
mask_rcnn/
├── custom_hooks/
│ ├── init.py
│ └── val_loss.py # Custom validation loss logging hook
├── analysis.py # Evaluation/analysis script
├── common.py # Utility functions
├── config.py # Python-side config loader
├── config.yml # YAML configuration
├── readme.md # Documentation
├── requirements.txt # Dependencies
└── train_code.py # Main training launcher
```


---

## ⚙️ Configuration

Edit the `config.yml` file to set your training preferences:

```yaml
config_path: './mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'
classes: ("Arms", "Rings")
load_from: './pretrained_models/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/your_model.pth'
work_dir: './results'
```
 - classes: Your dataset's class labels 
 - load_from: Path to the pretrained weights
 - work_dir: Folder to store logs and outputs

# ⚙️ Setup & 🚀 Training Guide

This document provides instructions for setting up the environment and running training for the Mask R-CNN project using MMDetection.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

If you're using this as part of a larger project:

```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```
### 2. Install Requirements
```bash
pip install -r requirements.txt
```
### 3. Download Pretrained Weights
```bash
https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth
```
Place the file as configured in config.yml under:
```bash
load_from: ./pretrained_models/your_model.pth
```
## 🚀 Training the Model
in train_code.py you must edit the ```setup_cfg```

### Major Configurations in `setup_cfg`

1.  **Dataset Configuration**  
    Sets paths for training, validation, and testing datasets including annotations and image folders. It also assigns the class names metadata consistently across all datasets.
    
2.  **Model Configuration**  
    Defines the number of output classes for the model’s bounding box head (here set to 2 classes).
    
3.  **Training Setup**  
    Controls training hyperparameters such as:
    
    -   Maximum number of epochs (`max_epochs=1` for quick tests)
        
    -   Batch size for all data loaders (50)
        
    -   Learning rate tuned for single GPU (`lr=0.005`)
        
    -   Validation and checkpoint intervals (every epoch)
        
4.  **Pretrained Weights & Output Directory**  
    Loads pretrained weights from the given path and sets the working directory for saving logs and checkpoints.
    
5.  **Evaluation Settings**  
    Specifies annotation files for the evaluator to enable accurate validation and test performance measurement.
    
6.  **Logging and Monitoring**  
    Adds hooks for:
    
    -   Logging progress and metrics (including TensorBoard support)
        
    -   Custom validation loss hook (`ValLoss`)
        
    -   Checkpoint saving and training timers
        
    -   Output log files and verbosity level configuration
        
7.  **Reproducibility and Hooks**  
    Sets a random seed for reproducibility (though non-deterministic mode is allowed) and establishes default training hooks that manage runtime behavior (logging, checkpointing, visualization, etc).
```bash
python train_code.py
```
This script will:

* Load the config from config.yml
* Initialize the dataset, model, and hooks
* Start the training loop
* Save logs and checkpoints to the directory specified in work_dir

## 🚀 Testing and Validation the Model

in ```analysis.py```

This code processes a set of images using a pre-trained object detection model from MMDetection. For each image, it:
-   Loads the image and runs inference to detect objects.
    
-   Extracts detected object information including class labels, confidence scores, bounding boxes, polygons, and centroids.
    
-   Saves all detection results to a JSON file.
    
-   Saves centroid and detection details to an Excel file.
    
-   Generates and saves images with detected objects visually overlaid.

It uses configuration and checkpoint files to initialize the model and processes all images in a specified directory. The output includes detailed detection data and visualizations for further analysis.

for Log Analysis you must join to 
https://mmdetection.readthedocs.io/en/dev/useful_tools.html