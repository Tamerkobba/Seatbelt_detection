
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from networkx.classes import non_edges
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay


df = pd.read_csv('../../classification_images/labels.csv')

df.head()

df = df.drop_duplicates()


def determine_class(row):
    if row['seatbelt'] == 1:
        return 'All Wearing Seatbelts'
    elif row['no_seatbelt'] == 1:
        return 'No Seatbelt'
    return 'Unknown'


df['Class'] = df.apply(determine_class, axis=1)


small_classes = df['Class'].value_counts()[df['Class'].value_counts() < 5].index
main_df = df[~df['Class'].isin(small_classes)]
_ = df[df['Class'].isin(small_classes)]

train_df, valid_df = train_test_split(
    main_df,
    test_size=0.15,
    stratify=main_df['Class'],
    random_state=645
)



label_mapping = {'All Wearing Seatbelts': 0, 'No Seatbelt': 1}
train_df['label'] = train_df.iloc[:, 1].map(label_mapping)
valid_df['label'] = valid_df.iloc[:, 1].map(label_mapping)

train_df = pd.get_dummies(train_df, columns=['label'], prefix='', prefix_sep='')
valid_df = pd.get_dummies(valid_df, columns=['label'], prefix='', prefix_sep='')



class ImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')  # Ensure RGB format

        if self.transform:
            image = self.transform(image)

        label = self.df.iloc[idx, 1]
        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = ImageDataset(train_df, '../../classification_images', transform=transform)
val_dataset = ImageDataset(valid_df, '../../classification_images', transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)



def metrics():
    def load_and_evaluate_model(checkpoint_path, train_loader, val_loader, num_classes=2, device='cuda'):
        """
        Load ResNet model from checkpoint and evaluate it
        """

        model = models.densenet121(weights=None)

        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:

            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        def collect_predictions(loader):
            y_true, y_pred, y_scores = [], [], []
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    scores = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)

                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    y_scores.extend(scores.cpu().numpy())
            return np.array(y_true), np.array(y_pred), np.array(y_scores)

        print("Collecting training predictions...")
        y_true_train, y_pred_train, y_scores_train = collect_predictions(train_loader)
        print("Collecting validation predictions...")
        y_true_val, y_pred_val, y_scores_val = collect_predictions(val_loader)

        def calculate_metrics(y_true, y_pred):
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_true, y_pred)
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix
            }

        train_metrics = calculate_metrics(y_true_train, y_pred_train)
        val_metrics = calculate_metrics(y_true_val, y_pred_val)

        print("\nTraining Metrics:")
        print(f"Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Precision: {train_metrics['precision']:.4f}")
        print(f"Recall: {train_metrics['recall']:.4f}")
        print(f"F1-Score: {train_metrics['f1']:.4f}")

        print("\nValidation Metrics:")
        print(f"Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}")
        print(f"Recall: {val_metrics['recall']:.4f}")
        print(f"F1-Score: {val_metrics['f1']:.4f}")

        def plot_confusion_matrix(conf_matrix, title):
            fig, ax = plt.subplots(figsize=(8, 6))  # Explicitly create a figure and axis
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
            disp.plot(cmap='Blues', ax=ax)  # Pass the axis to avoid creating a new one
            ax.set_title(title)  # Set the title for the axis
            plt.show()  # Show the plot

        plot_confusion_matrix(train_metrics['confusion_matrix'], 'Training Confusion Matrix')
        plot_confusion_matrix(val_metrics['confusion_matrix'], 'Validation Confusion Matrix')
        if y_scores_train.shape[1] == 2:
            def plot_roc_curve(y_true, y_scores, title):
                fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(title)
                plt.legend(loc='lower right')
                plt.show()
                return roc_auc

            train_auc = plot_roc_curve(y_true_train, y_scores_train, 'Training ROC Curve')
            val_auc = plot_roc_curve(y_true_val, y_scores_val, 'Validation ROC Curve')
        return model, train_metrics, val_metrics

    checkpoint_path = 'Best.pth.tar'

    model, train_metrics, val_metrics = load_and_evaluate_model(
        checkpoint_path=checkpoint_path,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


if __name__ == "__main__":
    metrics()