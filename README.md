# CNN Classifier

This project is a Convolutional Neural Network (CNN) classifier for images of cats, dogs, and pandas. The dataset is organized into the `dataset` directory, with three subdirectories: `cat`, `dog`, and `panda`, each containing 1000 images.

## Dataset Split

The dataset is split into training, validation, and test sets with the following proportions:

- Training Set: 70%
- Validation Set: 15%
- Test Set: 15%

## Training and Evaluation

We train the model for 45 epochs. At the end of each epoch, the following metrics are recorded and plotted:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

These metrics, plotted against the number of epochs, help us observe and judge the training performance of the model.

Finally, we evaluate the model's performance using the Test Loss and Test Accuracy.

## Usage

1. Clone or download this project.
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Run the main script:
    ```
    python main.py
    ```

## Results and Analysis
Training process metrics:

![image](https://github.com/liangchingyun/img-folder/blob/main/CNN-Classifier_result.png)


Final test results:\
Test Loss: 0.6178\
Test Accuracy: 71.78%