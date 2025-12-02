
# Similar Image Search and Generation

This project implements a deep learning pipeline for image similarity search and generation. The core of the project consists of two main components: an **Encoder** and a **Decoder**.

- The **Encoder** learns to create a compact representation (embedding) of an image. The goal is to produce similar embeddings for similar images and dissimilar embeddings for different images.
- The **Decoder** takes an embedding and generates an image that corresponds to that embedding.

## Goal

The primary goal of this project is to create a system that can:

1.  **Encode images** into a meaningful embedding space.
2.  **Compare image embeddings** to determine if they are similar.
3.  **Generate an image** from a given embedding.

The project is structured in three training phases:
1.  **Encoder Training**: The encoder is trained with an ArcFace loss to learn to discriminate between different classes of images.
2.  **Decoder Training**: The decoder is trained to reconstruct an image from its embedding.
3.  **Fine-tuning**: The encoder and decoder are trained together to ensure that the generated images have embeddings that are close to the original image's embedding.

## Project Structure

The project is organized into the following directories:

-   `My_Dataloader/`: Contains the data loading and preprocessing scripts.
-   `my_nn/`: Contains the neural network models (Encoder, Decoder, ArcFace).
-   `train/`: Contains the training scripts for the different training phases.
-   `checkpoints/`: Directory where the trained model checkpoints are saved.
-   `data/`: Default directory for the datasets.

## Datasets

The project is designed to be trained on a variety of image datasets. The `My_Dataloader` module handles the loading and preprocessing of the following datasets:

-   MNIST
-   FashionMNIST
-   Kuzushiji-MNIST
-   EMNIST
-   CIFAR-10
-   CIFAR-100
-   SVHN
-   STL-10

The dataloader automatically handles the differences in the APIs of these datasets and applies a consistent set of transformations. It also balances the datasets to ensure that each class is represented equally during training.

## Models

### Encoder

The encoder is a ResNet-style convolutional neural network that takes an image as input and outputs a 128-dimensional embedding vector. The embedding is L2-normalized to lie on a unit hypersphere, which is suitable for cosine similarity comparisons.

### Decoder

The decoder is a series of upsampling residual blocks that takes a 128-dimensional embedding vector as input and generates a 28x28 grayscale image.

### ArcFace

The ArcFace module is a metric learning layer that is used to train the encoder. It encourages the encoder to produce embeddings that are close together for images of the same class and far apart for images of different classes.

## Training Phases

The training process is managed by the `train_manager.py` script and is divided into three main phases:

### Phase 1: Encoder Training

- **Objective**: To train the encoder to produce discriminative embeddings.
- **Method**: The encoder is trained on a classification task using the ArcFace loss. This loss function encourages high intra-class similarity and high inter-class variance.
- **Validation**: The validation is a standard classification accuracy test. The training continues until a predefined accuracy threshold is met.

### Phase 2: Decoder Training

- **Objective**: To train the decoder to reconstruct an image from its embedding.
- **Method**: The encoder's weights are frozen, and the decoder is trained to minimize the mean squared error (MSE) between the original and the reconstructed image.
- **Validation**: The validation is based on the cosine similarity between the embedding of the original image and the embedding of the reconstructed image.

### Phase 3: Fine-Tuning

- **Objective**: To jointly optimize the encoder and decoder to improve the quality of the generated images and their embeddings.
- **Method**: Both the encoder and the decoder are trained simultaneously. The loss function is a combination of the reconstruction loss (MSE) and the classification loss (ArcFace).
- **Validation**: The validation includes both the classification accuracy and the reconstruction loss.

## How to Run

1.  **Installation**:
    It is recommended to use Python 3.10.
    Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Training**:
    To start the training process, run the `main.py` script:
    ```bash
    python main.py
    ```
    You can customize the training process by passing command-line arguments. For a full list of options, run:
    ```bash
    python main.py --help
    ```

3.  **Testing**:
    To test the trained models, run the `test.py` script:
    ```bash
    python test.py
    ```
    This will load the trained encoder and decoder, and for a few random images from the test set, it will show the original image and the reconstructed image, along with the cosine similarity between their embeddings.


