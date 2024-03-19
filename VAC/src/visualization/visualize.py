import matplotlib.pyplot as plt
import os, glob
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

def visualize_frames(data_dir):
    classes = os.listdir(data_dir)
    fig, axes = plt.subplots(len(classes), 6, figsize=(10, 6))  # 6 frames per class row

    for i, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        video_folders = os.listdir(class_path)
        frames_to_plot = []

        # Collecting frames from different videos of the class

        for video_folder in video_folders:
            video_frames_path = os.path.join(class_path, video_folder)
            frames = os.listdir(video_frames_path)
            frames = sorted(frames)[:10]  # Take the first 6 frames of each video

            for frame in frames:
                frame_path = os.path.join(video_frames_path, frame)
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                frames_to_plot.append(img)

            if len(frames_to_plot) >= 6:
                break

        # Plotting the frames
        for j in range(6):
            axes[i, j].imshow(frames_to_plot[j])
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(class_name)

    plt.tight_layout()
    plt.show()

def AUC(model_fitted):
    executed_epochs = len(model_fitted.history['loss'])

    auc = model_fitted.history['auc']
    val_auc = model_fitted.history['val_auc']

    epochs_range = range(executed_epochs)

    plt.figure(figsize=(18, 12))
    plt.plot(epochs_range, auc, label='Training AUC')
    plt.plot(epochs_range, val_auc, label='Validation AUC')
    plt.legend(loc='lower right')
    plt.title('Training and Test AUC')

def CF_matrix(true_classes, predicted_classes):
    cf = confusion_matrix(true_classes, predicted_classes)
    df_cm = pd.DataFrame(cf)
    plt.figure(figsize = (10,10))
    sns.heatmap(df_cm, annot=True)