import numpy as np
import cv2
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# Dice metric and loss (same as training)
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-6) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-6)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Data generator (same as training)
def data_generator(txt_file, image_size=(256, 256), batch_size=8):
    with open(txt_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    image_mask_pairs = [(line.split(',')[0], line.split(',')[1]) for line in lines]

    for i in range(0, len(image_mask_pairs), batch_size):
        batch_pairs = image_mask_pairs[i:i + batch_size]
        batch_imgs, batch_masks = [], []

        for img_path, mask_path in batch_pairs:
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            img = img / 255.0

            mask = Image.open(mask_path).convert('L')
            mask = mask.resize(image_size, resample=Image.NEAREST)
            mask = np.array(mask)
            mask = (mask > 0).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            batch_imgs.append(img)
            batch_masks.append(mask)

        yield np.array(batch_imgs), np.array(batch_masks)

# Load model and evaluate on validation set
def test_model(model_path, val_txt, image_size=(256, 256), batch_size=4):
    # Load model with custom dice loss and dice metric
    model = load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})

    val_gen = data_generator(val_txt, image_size=image_size, batch_size=batch_size)
    val_steps = sum(1 for _ in open(val_txt)) // batch_size

    results = model.evaluate(val_gen, steps=val_steps)
    val_loss, val_dice, val_acc = results

    print(f"Test Results for model '{model_path}':")
    print(f" - Loss: {val_loss:.4f}")
    print(f" - Dice Coefficient: {val_dice:.4f}")
    print(f" - Accuracy: {val_acc:.4f}")

# Example usage:
if __name__ == "__main__":
    fold = 0  # Choose which fold model to test
    model_path = f"unet_fold{fold}_best.h5"  # unet_fold3_best.h5 is the best one
    val_txt = f"folds/val_fold{fold}.txt"
    test_model(model_path, val_txt)
