# Face Recognition
It is repository for AI internship project at Internship Studio. 

Based on the provided code snippet, here's an overview of what the project aims to achieve and how it accomplishes it:

### Project Overview:
The project appears to be focused on implementing a face recognition system using machine learning techniques. Here's a breakdown of the components and their roles:

1. **Importing Libraries**:
   - `import os`, `import cv2`, `import numpy as np`, `import matplotlib.pyplot as plt`: These lines import necessary libraries for file operations (`os`), image processing (`cv2`), numerical operations (`numpy`), and visualization (`matplotlib`).

2. **Helper Function: `plot_gallery`**:
   - This function is designed to plot a gallery of images (portraits in this case), arranged in a grid format (`n_row` rows and `n_col` columns). It adjusts subplot spacing and visual properties for displaying images.

3. **Data Loading and Preprocessing**:
   - **Directory Setup**: `dir_name` specifies the path where face images are stored.
   - **Image Loading and Resizing**: Images are loaded using `cv2.imread`, converted to grayscale (`cv2.cvtColor`), resized to a standard dimension (`300x300`), and flattened into 1D arrays (`v`).
   - **Labels and Names**: Each image is associated with a label (`person_id`) and a name (`person_name`), stored in `y` and `target_names`, respectively.

4. **Data Preparation**:
   - **Convert to Numpy Arrays**: Convert lists `X`, `y`, and `target_names` into `numpy` arrays for further processing.
   - **Print Data Statistics**: Display the shape of arrays (`y.shape`, `X.shape`, `target_names.shape`) and the total number of samples (`n_samples`).

5. **Splitting Data**: 
   - **Train-Test Split**: Split the data into training and testing sets using `train_test_split` from `sklearn.model_selection`.

6. **Dimensionality Reduction with PCA**:
   - **PCA Application**: Apply Principal Component Analysis (`PCA`) to reduce the dimensionality of the feature space (`X_train`). This extracts the most significant features (`eigenfaces`) that represent the variability in face images.

7. **Linear Discriminant Analysis (LDA)**:
   - **LDA Transformation**: Use Linear Discriminant Analysis (`LDA`) to further enhance the discriminatory power of the features extracted by PCA. This step projects the data into a lower-dimensional space (`X_train_lda`, `X_test_lda`).

8. **Training a Machine Learning Model**:
   - **MLPClassifier**: Train a Multi-Layer Perceptron (`MLPClassifier`) using the transformed data (`X_train_lda`, `y_train`) to learn patterns and classify faces.

9. **Model Evaluation**:
   - **Prediction and Accuracy**: Predict labels (`y_pred`) for test data (`X_test_lda`). Compute accuracy by comparing predicted labels with true labels (`y_test`). Print model weights (`clf.coefs_`) and display prediction results (`plot_gallery`).

### Purpose of `import os`:
- `os` module is imported to handle directory operations (`os.listdir`, `os.path.join`). It facilitates navigation through directories (`dir_name`, `dir_path`) to access image files stored in a structured manner.
  
### Project Aim:
- **Face Recognition**: The project's primary goal is to develop a system capable of recognizing faces from images using machine learning techniques. This involves preprocessing images, extracting meaningful features using PCA and LDA, training a classifier (MLP), and evaluating its accuracy.

By integrating these steps, the project aims to automate the process of identifying individuals from face images, potentially useful in security systems, surveillance, and various other applications requiring reliable identification mechanisms.
