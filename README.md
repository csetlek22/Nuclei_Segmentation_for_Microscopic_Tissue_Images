# Nuclei Segmentation with U-Net

This repository contains code to train and evaluate a U-Net model for nuclei instance segmentation on H&E-stained histological images.

---

## 📚 Dataset

We use the [NuInsSeg dataset](https://www.kaggle.com/datasets/ipateam/nuinsseg/data):

> @article{mahbod2023nuinsseg,  
> &nbsp;&nbsp;title={NuInsSeg: A Fully Annotated Dataset for Nuclei Instance Segmentation in H\&E-Stained Histological Images},  
> &nbsp;&nbsp;author={Mahbod, Amirreza and Polak, Christine and Feldmann, Katharina and Khan, Rumsha and Gelles, Katharina and Dorffner, Georg and Woitek, Ramona and   
> &nbsp;&nbsp;&nbsp;&nbsp;Hatamikia, Sepideh and Ellinger, Isabella},  
> &nbsp;&nbsp;journal={arXiv preprint arXiv:2308.01760},  
> &nbsp;&nbsp;year={2023}  
> }

You can download the dataset here:  
[https://www.kaggle.com/datasets/ipateam/nuinsseg/data](https://www.kaggle.com/datasets/ipateam/nuinsseg/data)

---

## 📂 Project Structure

- `all_pairs_path.py`: Generates train/val/test splits from your dataset.
- `creating_fold5.py`: Creates 5-fold cross-validation splits.
- `train.py`: Training script that runs 5-fold cross-validation with U-Net.
- `test.py`: Script to evaluate saved fold models on validation sets.

---

## ⚙️ Setup Environment

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/nuclei-segmentation.git
   cd nuclei-segmentation
````

2. Install dependencies (preferably in a virtual environment):

   ```bash
   pip install tensorflow numpy opencv-python pillow scikit-learn matplotlib
   ```

---

## 🛠️ Prepare Dataset Splits

1. Place all image and mask file paths in a text file called `image_mask_pairs.txt` in the format:

   ```
   /path/to/image1.png,/path/to/mask1.png
   /path/to/image2.png,/path/to/mask2.png
   ...
   ```

2. Run `all_pairs_path.py` to generate `folds_collab/train.txt`, `val.txt`, and `test.txt` splits:

   ```bash
   python all_pairs_path.py
   ```

3. Run `creating_fold5.py` to create 5-fold cross-validation splits inside the `folds` directory:

   ```bash
   python creating_fold5.py
   ```

---

## 🚀 Training

Run the training script to train U-Net models on 5 folds with Dice loss:

```bash
python train.py
```

* Trains 5 models, each on different train/val splits.
* Saves best weights per fold as `unet_fold{fold}_best.h5`.
* Prints fold-wise Dice coefficient, accuracy, and loss after training.

---

## 🧪 Testing / Evaluation

Evaluate a trained fold model on its validation split using:

```bash
python test.py
```

* By default, evaluates fold 0. You can modify the fold number inside `test.py`:

```python
fold = 0
model_path = f"unet_fold{fold}_best.h5"
val_txt = f"folds/val_fold{fold}.txt"
```

* Prints loss, Dice coefficient, and accuracy.

---

## 📁 Download Pretrained Models

If you want to skip training, download pretrained weights here:
[Google Drive Link](https://drive.google.com/drive/folders/1X34AUJ3c7bktOJ2s4yEFPOhIlISiR07I?usp=sharing)

Place the `.h5` files in the root directory before running `test.py`.

---

## 📌 Notes

* Make sure image and mask paths are correct and accessible.
* Adjust `batch_size`, `image_size`, and `epochs` in `train.py` and `test.py` to suit your hardware.
* This repo uses TensorFlow 2.x and Keras API.
* The Dice coefficient is the main metric for segmentation quality.

---

## 📝 Citation

If you use this code or dataset, please cite the NuInsSeg paper:

```
@article{mahbod2023nuinsseg,
  title={NuInsSeg: A Fully Annotated Dataset for Nuclei Instance Segmentation in H\&E-Stained Histological Images},
  author={Mahbod, Amirreza and Polak, Christine and Feldmann, Katharina and Khan, Rumsha and Gelles, Katharina and Dorffner, Georg and Woitek, Ramona and 
          Hatamikia, Sepideh and Ellinger, Isabella},
  journal={arXiv preprint arXiv:2308.01760},
  year={2023}
}
```

---

Feel free to ask if you want me to add CLI arguments or a more detailed tutorial section!
