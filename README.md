# Ship Detection System
Detection Ships on satellite images with Neural Network, Keras, Unet Architecture, Dice Score, Matplotlib, Python.
** by Serhii Spitsyn**
---
My LinkedIn [LinkedIn](https://www.linkedin.com/in/serhii-spitsyn-ba98b82a9/)

![Ship Detection](https://github.com/ShamansIT/Airbus_Ship_Detection_Kaggle/blob/main/Resource/ship_detect.png)

## Overview
This Ship Detection System is designed to identify and locate ships on satellite images rapidly. Utilizing advanced machine learning algorithms, this program aims to tackle the challenging task of distinguishing ships from various natural and artificial objects within the complex visual context of satellite imagery. The primary application of this system spans across maritime navigation, environmental monitoring, maritime security, and resource exploration.
---

## Features
Ship Detection: Utilizes Convolutional Neural Networks (CNNs) to accurately detect ships of various sizes and types.
Image Preprocessing: Implements techniques to enhance image quality and normalize conditions for optimal detection performance.
Data Augmentation: Increases the robustness of the model by expanding the training dataset with modified versions of the original images to simulate a variety of scenarios.
Object Classification: Distinguishes ships from other objects, such as waves, clouds, and coastal features, using advanced classification algorithms.
Scalability: Designed to efficiently process large datasets of satellite images with high throughput and accuracy.
#### Prerequisites: Python 3.3 or higher
#### Dependencies: TensorFlow, NumPy, OpenCV, and other required libraries listed in requirements.txt.
---

## Model Training
The system comes with a pre-trained model. However, you can retrain the model with your dataset for improved accuracy or adaptation to specific types of ships by following the instructions in the training/ directory.

## Stage developing:
---
### Stage 1: Files and Annotations Verification
Check the dataset files and their corresponding annotations to ensure that satellite images and ship location masks are correctly aligned and formatted. This involves verifying the integrity and consistency of image files, as well as ensuring that annotations accurately reflect ship positions on the images. This step is crucial for preparing a clean and reliable dataset for subsequent processing and analysis stages.

### Stage 2: Data Loading and Preprocessing
Download the dataset with satellite images and annotations (masks) indicating the location of ships.
Divide the data into training, validation, and test sets.
Apply preprocessing to the data: scale pixel values, augment the data (to increase the diversity of the training set), and convert masks to the appropriate format.

### Stage 3: Creating a Model with U-Net Architecture
Define the U-Net architecture using tf.keras. This model should include contraction blocks (to capture context) and expansion blocks (for precise localization).
Compile the model, selecting an optimizer (e.g., Adam) and a loss function. For segmentation tasks, a combination of Dice score-based loss and cross-entropy is often used.

### Stage 4: Model Training
Train the model on the training dataset, using the validation set to monitor the training process.
Use tf.keras callbacks such as ModelCheckpoint to save the best model and EarlyStopping to prevent overfitting.

### Stage 5: Model Evaluation
Evaluate the model's performance on the test dataset using the Dice score and other relevant metrics (e.g., accuracy, recall).
Visualize the model's predictions on test images for qualitative assessment.

### Stage 6: Fine-tuning and Optimization
Based on the testing results, you may conduct further fine-tuning of hyperparameters, model structure, or the data preprocessing process.
Repeat training and evaluation until satisfactory results are achieved.

### Stage 7: Deployment
Prepare the model for deployment: save the trained model and develop infrastructure for its use in real-world conditions.
Integrate the model into an application or service, where it will be used to detect ships on satellite images.
