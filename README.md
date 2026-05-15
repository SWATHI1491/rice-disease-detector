# Rice Disease Detection using CNN

This project detects rice leaf diseases using a Convolutional Neural Network trained on the Kaggle Rice Leaf Diseases dataset.

Custom CNN trained on Kaggle's Rice Leaf Diseases dataset to classify three rice diseases — Bacterial Leaf Blight, Brown Spot, and Leaf Smut — from leaf imagery. Built a full pipeline: image preprocessing (224×224),  3-block Conv2D architecture with Dropout regularization, 100-epoch training, confusion matrix evaluation, and an automated fertilizer/treatment recommendation engine per predicted class.
## Dataset

The dataset is downloaded directly from Kaggle using KaggleHub:

```python
kagglehub.dataset_download("vbookshelf/rice-leaf-diseases")
