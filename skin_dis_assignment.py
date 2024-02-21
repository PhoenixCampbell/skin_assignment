import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
layers = tf.keras.layers
Sequential = tf.keras.models.Sequential


def load_and_preprocess_data(data_dir, img_height=128, img_width=128, batch_size=16):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    return train_ds, val_ds


def build_and_train_model(train_ds, val_ds, epochs=50):
    num_classes = len(train_ds.class_names)
    model = Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return history


def plot_training_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()


def plot_learning_curve(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curve: Loss")
    plt.legend()
    plt.show()


def plot_roc_curve(model, test_ds, y_true):
    y_pred_prob = model.predict(test_ds)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()


def plot_accuracy_vs_epochs(history):
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.legend()
    plt.show()


def evaluate_model(model, test_ds, class_names):
    test_loss, test_accuracy = model.evaluate(test_ds)
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

    y_pred = np.argmax(model.predict(test_ds), axis=1)
    y_true = test_ds.labels

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred, class_names)


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm = (
        cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    )  # Normalize confusion matrix

    plt.figure(figsize=(8, 8))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


if __name__ == "__main__":
    data_dir = "./Images"
    train_ds, val_ds = load_and_preprocess_data(data_dir)
    img_height = 128
    img_width = 128

    history = build_and_train_model(
        train_ds,
        val_ds,
    )
    model = history.model
    plot_learning_curve(history)
    plot_training_history(history)
    y_true = np.concatenate([y for x, y in val_ds], axis=0)
    # plot_roc_curve(model, val_ds, y_true)
    plot_accuracy_vs_epochs(history)
