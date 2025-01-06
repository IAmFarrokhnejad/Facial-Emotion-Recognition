# Facial-Emotion-Recognition
 

This repository contains code for training and validating image classification models using three popular architectures: EfficientNet, MobileNetV2, and ShuffleNet. The code supports handling imbalanced datasets, tracks training metrics, and generates classification reports and confusion matrices.

## Features

- **Architectures Supported**:
  - EfficientNet-B0
  - MobileNetV2
  - ShuffleNetV2
- **Handles Imbalanced Data**: Weighted sampling ensures balanced training.
- **Performance Visualization**: Generates classification reports and confusion matrices.
- **Customizable**: Easily modify training parameters and dataset configurations.

---

## Requirements

### Libraries
Ensure the following Python libraries are installed:
- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### Hardware
- A CUDA-compatible GPU is recommended for efficient training.

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/IAmFarrokhnejad/Facial-Emotion-Recognition
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision matplotlib seaborn scikit-learn
   ```

3. **Prepare Your Dataset**:
   - The dataset should follow the directory structure:
     ```
     dataset/
     ├── train/
     │   ├── class_1/
     │   ├── class_2/
     │   └── ...
     └── test/
         ├── class_1/
         ├── class_2/
         └── ...
     ```
   - Update the `data_dir` variable in the script to point to your dataset path.

4. **Configure Training Parameters**:
   Modify the following constants in the script as per your requirements:
   ```python
   BATCH_SIZE = 32
   EPOCHS = 25
   IMAGE_SIZE = 48
   LEARNING_RATE = 0.001
   ```

---

## Running the Code

To start training and validation:
```bash
python main.py
```

The script will:
1. Train and validate models sequentially for EfficientNet, MobileNetV2, and ShuffleNet.
2. Save the classification report and confusion matrix for each model in the specified directory.

---

## Output Files

- **Logs**: Classification reports and confusion matrices are saved in the folder you specify.
- **Visualizations**: Confusion matrices are displayed as heatmaps during execution.

---

## Function Details

### `get_model(architecture, num_classes)`
Creates a specified model with adjusted layers for grayscale images and a custom number of output classes.

### `train_and_validate(model, criterion, optimizer, dataloaders, architecture, num_epochs)`
Handles the training and validation phases, printing metrics and generating outputs.

### `plotConfusionMatrix(cm, dataset_name, normalize=False, class_labels=None)`
Plots confusion matrices using Seaborn for better interpretability.

---

## Future Improvements

- Further algorithm optimization.
- Add support for additional model architectures.


---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

[Morteza Farrokhnejad](https://github.com/IAmFarrokhnejad)

---

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework.
- The creators of EfficientNet, MobileNetV2, and ShuffleNet for providing pretrained models.
- Seaborn and Matplotlib for visualization.