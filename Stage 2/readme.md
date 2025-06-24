## Stage 2: Model Validation

### Overview

In this stage, we validate the model’s results by comparing detected fixations on annotated Areas of Interest (AOIs) with manual and automatic (Tobii) hit results.

### Step 1: Process Video Frames

-   The video frames used for validation must be processed using the model output from the selected epoch.
    
-   In our case, **epoch 49** is used.
    
-   Refer to the processing script in:  
    **`Stage 1/analysis.py`**
    

### Step 2: AOI Detection

-   The model output contains detected AOIs labeled as **"Arms"** and **"Rings"**, which correspond to **"graspers"** and **"objects"**, respectively.
    
-   An example of the model’s annotation output can be found in:  
    **`Stage 2/Stage 2 Data/result_Adult.json`**
    

### Step 3: Fixation Hit Detection

-   To compute fixation hits on graspers and objects, use:
    
    -   **`result_Adult.json`** (model-predicted AOIs)
        
    -   **`Fixation_Adult.xlsx`** (Tobii fixation data)
        
-   Run the script:  
    **`Stage 2/Stage 2 Code/Hit Rings and Arms.py`**
    

### Step 4: Result Comparison

-   Compare the fixation hits obtained using the model with the ground truth:
    
    -   **Manual hit results**
        
    -   **Auto hit results (Tobii-generated)**
        
-   Reference comparison data file:  
    **`Stage 2/Stage 2 Data/Gaze Data/Compare Results.xls`**
