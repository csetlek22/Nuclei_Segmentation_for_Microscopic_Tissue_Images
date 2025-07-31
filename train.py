import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras import layers, models, optimizers, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold


# =============== Data Generator ===============
def data_generator(txt_file, image_size=(256, 256), batch_size=8):
    with open(txt_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    image_mask_pairs = [(line.split(',')[0], line.split(',')[1]) for line in lines]

    while True:
        np.random.shuffle(image_mask_pairs)
        for i in range(0, len(image_mask_pairs), batch_size):
            batch_pairs = image_mask_pairs[i:i + batch_size]
            batch_imgs = []
            batch_masks = []

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


# =============== Dice Metric / Loss ===============
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-6) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-6)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# =============== U-Net Model ===============
def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        return x

    def encoder_block(x, filters):
        f = conv_block(x, filters)
        p = layers.MaxPooling2D((2, 2))(f)
        return f, p

    def decoder_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)
        return x

    f1, p1 = encoder_block(inputs, 64)
    f2, p2 = encoder_block(p1, 128)
    f3, p3 = encoder_block(p2, 256)
    f4, p4 = encoder_block(p3, 512)

    bottleneck = conv_block(p4, 1024)

    d1 = decoder_block(bottleneck, f4, 512)
    d2 = decoder_block(d1, f3, 256)
    d3 = decoder_block(d2, f2, 128)
    d4 = decoder_block(d3, f1, 64)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)

    model = models.Model(inputs, outputs)
    return model


# =============== 5-Fold Cross Validation ===============
def run_kfold_training(num_folds=5, image_size=(256, 256), batch_size=4, epochs=10):
    all_metrics = []

    for fold in range(num_folds):
        print(f"\n=== Training Fold {fold} ===")
        train_txt = f"folds/train_fold{fold}.txt"
        val_txt = f"folds/val_fold{fold}.txt"

        # Generators
        train_gen = data_generator(train_txt, image_size=image_size, batch_size=batch_size)
        val_gen = data_generator(val_txt, image_size=image_size, batch_size=batch_size)

        # Count lines to determine steps
        train_steps = sum(1 for _ in open(train_txt)) // batch_size
        val_steps = sum(1 for _ in open(val_txt)) // batch_size

        # Model
        model = build_unet(input_shape=(image_size[0], image_size[1], 3))
        model.compile(optimizer=optimizers.Adam(1e-4), loss=dice_loss, metrics=[dice_coef, 'accuracy'])

        # Save best model per fold
        checkpoint_path = f"unet_fold{fold}_best.h5"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

        # Training
        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=[checkpoint]
        )

        # Load best model and evaluate on val set
        model.load_weights(checkpoint_path)
        val_loss, dice, acc = model.evaluate(val_gen, steps=val_steps)
        print(f"Fold {fold} Final Dice: {dice:.4f}, Accuracy: {acc:.4f}, Loss: {val_loss:.4f}")
        all_metrics.append((dice, acc, val_loss))

    # Summary
    print("\n=== Cross-Validation Summary ===")
    for i, (dice, acc, loss) in enumerate(all_metrics):
        print(f"Fold {i}: Dice={dice:.4f}, Acc={acc:.4f}, Loss={loss:.4f}")


run_kfold_training(num_folds=5, image_size=(256, 256), batch_size=4, epochs=10)
