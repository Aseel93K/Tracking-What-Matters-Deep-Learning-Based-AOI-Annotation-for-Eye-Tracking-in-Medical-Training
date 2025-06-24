# Tracking-What-Matters-Deep-Learning-Based-AOI-Annotation-for-Eye-Tracking-in-Medical-Training

## Overview

This repository is intended for researchers and authors who wish to apply a two-stage process involving **Mask R-CNN development and validation**, integrating eye-tracking data with dynamically detected Areas of Interest (AOIs).

Although the workflow is demonstrated using the **peg transfer task**, the same procedure can be adapted to other tasks involving video-based AOI detection and gaze data integration.

----------

## Workflow

The process is divided into two main stages:

### **Stage 1: Model Development**

-   Train a **Mask R-CNN** model to detect relevant AOIs (e.g., graspers and objects).
    
-   The AOIs are used as dynamic regions for gaze mapping.
    
-   Refer to the `Stage 1` folder for training scripts and configuration.
    

### **Stage 2: Model Validation**

-   Validate the detected AOIs by comparing fixation hits with manual and automatically generated hits from Tobii software.
    
-   This includes:
    
    -   Processing model outputs
        
    -   Mapping fixations from Tobii data onto dynamic AOIs
        
    -   Running comparison analyses
        

Refer to the corresponding scripts and files under the `Stage 2` folder.

----------

## How to Use

1.  Begin with **Stage 1** to train the Mask R-CNN model on your dataset.
    
2.  Proceed to **Stage 2** to validate the model predictions using fixation data.
    

Make sure to follow the file paths and script instructions provided in each stage directory.
