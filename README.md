# CNN Image Classification Pipeline

A compact, end to end demonstration of an image classification pipeline using a Convolutional Neural Network in TensorFlow/Keras. The notebook walks through every stage: data loading, preprocessing, visualization, model building, training, evaluation and metric reporting. It is built to be adapted to any folder-per-class image dataset.

A chest X-ray dataset (3 classes) is used as the example data, but the pipeline is dataset agnostic.

## Pipeline Stages

1. Mount Google Drive and configure dataset paths.
2. Load images with `ImageDataGenerator` and preview a 4x4 grid of samples with labels.
3. Build `train\_ds`, `val\_ds` and `test\_ds` with `image\_dataset\_from\_directory`, using a 20% validation split.
4. Define a 4 block CNN.
5. Train with Adam and categorical crossentropy.
6. Evaluate on the test set.
7. Plot accuracy and loss curves for training and validation.
8. Generate a confusion matrix and report weighted precision, recall and F1 score.

## Model Architecture

|Layer|Details|
|-|-|
|Input|256 x 256 x 3|
|Conv2D + MaxPooling|32 filters, 3x3, ReLU, 2x2 pool|
|Conv2D + MaxPooling|64 filters, 3x3, ReLU, 2x2 pool|
|Conv2D + MaxPooling|128 filters, 3x3, ReLU, 2x2 pool|
|Conv2D + MaxPooling|256 filters, 3x3, ReLU, 2x2 pool|
|Flatten||
|Dense|128 units, ReLU|
|Dropout|0.5|
|Dense (output)|3 units, Softmax|

## Training Configuration

|Parameter|Value|
|-|-|
|Image size|256 x 256|
|Batch size|32|
|Optimizer|Adam (learning rate 0.0001)|
|Loss|Categorical crossentropy|
|Epochs|50|
|Validation split|20% of the training set|

## Dataset Layout

Any dataset organized with one folder per class works:

```
dataset/
├── train/
│   ├── class\_a/
│   ├── class\_b/
│   └── class\_c/
└── test/
    ├── class\_a/
    ├── class\_b/
    └── class\_c/
```

To use a different dataset, update the path variables in the first cell and set the output layer units to your number of classes.

## Requirements

* Python 3.8+
* TensorFlow 2.x
* NumPy
* Pandas
* Matplotlib
* Seaborn
* scikit-learn

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Upload your dataset to Google Drive.
2. Open `cnn\_image\_classification\_pipeline.ipynb` in Google Colab.
3. Edit the dataset paths in the first cell.
4. Run all cells.

## Results

Add your final numbers here after training.

|Metric|Value|
|-|-|
|Test accuracy||
|Precision (weighted)||
|Recall (weighted)||
|F1 score (weighted)||

## Known Limitations

1. The model receives raw 0 to 255 pixel values. Adding a `Rescaling(1./255)` layer at the start of the model usually improves convergence.
2. `test\_ds` is built with `validation\_split=0.2` and `subset="training"`, so only 80% of the test images are evaluated. Remove the split arguments to evaluate the full test set.
3. The `ImageDataGenerator` section (64 x 64) is used only for sample visualization and is separate from the training pipeline.
4. No data augmentation or transfer learning is applied.

## 

