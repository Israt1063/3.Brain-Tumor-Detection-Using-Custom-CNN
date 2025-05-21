

# Brain Tumor Detection Using Custom CNN

## Overview

This project implements a **custom Convolutional Neural Network (CNN)** classifier to detect brain tumors from MRI images. The dataset contains two classes: **`yes`** (tumor present) and **`no`** (no tumor).

The model is trained and validated on brain MRI images using PyTorch, with data augmentation and normalization applied for better generalization.

---

## Dataset

The dataset used for this project is publicly available:

**Brain MRI Images for Brain Tumor Detection**
[Download Link (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

After downloading, unzip and arrange the data in the following structure:

```
brain_tumor_data/
    brain_tumor_dataset/
        no/      # MRI images without tumor
        yes/     # MRI images with tumor
```

---

## Requirements

* Python 3.x
* PyTorch
* torchvision
* PIL (Pillow)
* CUDA (optional, for GPU training)

Install dependencies:

```bash
pip install torch torchvision pillow
```

---

## Usage

1. **Download and prepare the dataset** in the above folder structure.
2. **Set up data transforms and dataloaders**.
3. **Define the CNN model** (`BrainTumorCNN`).
4. **Train the model** using the provided training loop.
5. **Evaluate the model** on validation/test sets.

---

## Code Outline

* `BrainTumorDataset`: Custom PyTorch dataset to load MRI images.
* `train_transforms` & `val_transforms`: Data augmentation and normalization.
* `BrainTumorCNN`: Custom CNN architecture.
* `train_classifier()`: Function for model training and validation.

---

## Training Results

* The model achieved an average validation accuracy of around **77.7%** over 10 epochs.
* Validation accuracy improved from \~45% to \~85% during training.

---

## Next Steps

* Test the model on unseen test data.
* Experiment with transfer learning using pretrained models (ResNet, EfficientNet).
* Add more data augmentations.
* Implement early stopping and learning rate scheduling.

---

## Save and Load Model

Save trained weights:

```python
torch.save(model.state_dict(), "brain_tumor_cnn.pth")
```

Load model weights:

```python
model = BrainTumorCNN(num_classes=2)
model.load_state_dict(torch.load("brain_tumor_cnn.pth"))
model.eval()
```

---

## License

MIT License 

