```markdown
+---------------------+
|  Data Loading Block |
+---------------------+
            |
            v
+---------------------+
|      PCA Block      |
+---------------------+
            |
            v
+---------------------+
|      LDA Block      |
+---------------------+
            |
            v
+-----------------------------+
|    MLP Classifier Block     |
+-----------------------------+
            |
            v
+-----------------------------+
|    Visualization Block      |
+-----------------------------+
```
In the above flowchart, we have defined different blocks for building a working model.

## Block 1: Import Libraries

In the import libraries block, we will import the necessary libraries to perform tasks, as libraries provide functions to perform operations.
In this model, we are using:
1. numpy - for numerical operations
2. opencv - for image processing
3. os - for operating system operations
4. matplotlib - for visualizing data and output
5. sklearn - for scientific arithmetic operations

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
```

## Block 2: Data Loading Block

In the data loading block, we are loading our data from a dataset located on the local system drive. We are preprocessing the data as follows:
1. Define variables.
2. Fetch images with `dir_path`.
3. Read and represent images in grayscale for the regression model to understand the input data.
4. Resize images with height and width parameters.
5. Increase and add samples of the current image and `person_id`, then process the next image with the append function.
6. Store values in arrays: `X` for image data and `y` for label data.
7. Reshape the two-dimensional array into a single-dimensional array.
8. Print the values.

```python
dir_name = r"C:\\Users\\Sparx\\Downloads\\dataset\\dataset\\faces"
y = []
X = []
target_names = []
person_id = 0
h = w = 300
n_samples = 0
class_names = []

for person_name in os.listdir(dir_name):
    dir_path = os.path.join(dir_name, person_name)
    class_names.append(person_name)
    for image_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image_name)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, (h, w))
        v = resized_image.flatten()
        X.append(v)
        n_samples += 1
        y.append(person_id)
        target_names.append(person_name)
    person_id += 1

y = np.array(y)
X = np.array(X)
target_names = np.array(target_names)
n_features = X.shape[1]

print(y.shape, X.shape, target_names.shape)
print("Number of samples:", n_samples)
```

## Block 3: PCA Block - Principal Component Analysis

1. Import libraries from `sklearn`.
2. Split the data into training and testing datasets.
3. `X` will have image data and `y` will have label data.
4. `test_size=0.25` means 25% of the data is reserved for testing. `random_state=42` sets the seed for reproducibility.
5. Represent `X` and `y` for training and testing.
6. In the PCA model, process and reshape the data with the `fit` and `reshape` functions.
7. `eigenfaces` are the reshaped principal components.
8. Print the reshaped training and testing data.

```python
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

n_components = 150
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape, X_test_pca.shape)
```

## Block 4: LDA Block - Linear Discriminant Analysis

1. Import `LinearDiscriminantAnalysis` from `sklearn.discriminant_analysis`.
2. Create an instance of `LinearDiscriminantAnalysis`.
3. Fit the LDA model using `X_train_pca` and `y_train`.
4. Transform `X_train_pca` and `X_test_pca` using the LDA model.
5. Print "LDA transformation done..." after transforming the data.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_pca, y_train)

X_train_lda = lda.transform(X_train_pca)
X_test_lda = lda.transform(X_test_pca)
print("LDA transformation done...")
```

## Block 5: MLP Classifier Block

1. Import `MLPClassifier` from `sklearn.neural_network`.
2. Create an instance of the MLP classifier.
   - `random_state=1` sets the seed for reproducibility.
   - `hidden_layer_sizes=(10, 10)` specifies the architecture of the neural network with two hidden layers, each containing 10 neurons.
   - `max_iter=1000` sets the maximum number of iterations for training the neural network to 1000.
   - `verbose=True` enables verbose output during training.
3. Train the MLP classifier using the LDA-transformed training data.
4. Print the shapes of the weight matrices in the neural network.
5. Initialize empty lists to store predicted class labels and probabilities for the test data.
6. For each test face, predict the class and store the predicted class ID and probability.
7. Calculate and print the accuracy of the model.

```python
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(random_state=1, hidden_layer_sizes=(10, 10), max_iter=1000, verbose=True)
clf.fit(X_train_lda, y_train)

print("Model Weights:")
model_info = [coef.shape for coef in clf.coefs_]
print(model_info)

y_pred = []
y_prob = []

for test_face in X_test_lda:
    prob = clf.predict_proba([test_face])[0]
    class_id = np.where(prob == np.max(prob))[0][0]
    y_pred.append(class_id)
    y_prob.append(np.max(prob))

y_pred = np.array(y_pred)
prediction_titles = []
true_positive = 0

for i in range(y_pred.shape[0]):
    true_name = class_names[y_test[i]]
    pred_name = class_names[y_pred[i]]
    result = 'pred: %s, pr: %s \ntrue: %s' % (pred_name, str(y_prob[i])[:3], true_name)
    prediction_titles.append(result)
    if true_name == pred_name:
        true_positive += 1

print("Accuracy:", true_positive * 100 / y_pred.shape[0])
```

## Block 6: Visualization Block

1. Import `matplotlib.pyplot` for plotting images and creating visualizations.
2. Define the `plot_gallery` function to create a grid of subplots to display images with their titles, adjusting layout and formatting.
3. Generate titles for eigenfaces and display them using `plot_gallery`, followed by `plt.show()` to render the plot.
4. Display test faces along with prediction titles using `plot_gallery`, followed by `plt.show()` to render the plot.

```python
import matplotlib.pyplot as plt

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()

plot_gallery(X_test, prediction_titles, h, w)
plt.show()
```
```
