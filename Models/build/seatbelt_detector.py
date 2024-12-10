import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.models import densenet121, DenseNet121_Weights

df=pd.read_csv('../../../classification_images/labels.csv')

df = df.drop_duplicates()

def determine_class(row):
    if row['seatbelt'] == 1:
        return 'All Wearing Seatbelts'
    elif row['no_seatbelt'] == 1:
        return 'No Seatbelt'
    return 'Unknown'

df['Class'] = df.apply(determine_class, axis=1)
train_df, valid_df = train_test_split(
    main_df,
    test_size=0.2,
    stratify=main_df['Class'],
    random_state=420
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

        # Check for existence of images
        self.valid_indices = self.df[self.df.iloc[:, 0].apply(
            lambda img: os.path.exists(os.path.join(self.root_dir, img))
        )].index

        # Filter the dataframe to only valid images
        self.df = self.df.loc[self.valid_indices]

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
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Create datasets
train_dataset = ImageDataset(train_df, '../../../classification_images', transform=transform)
val_dataset = ImageDataset(valid_df, '../../../classification_images', transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)


weights = DenseNet121_Weights.IMAGENET1K_V1
model = densenet121(weights=weights)
numb_ftrs = model.classifier.in_features
model.classifier = nn.Linear(numb_ftrs, 2)
# Update the criterion to include class weights

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.7, weight_decay=0.001)
#optimizer= optim.Adam(model.parameters(), lr=0.0001,weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=50
)
model.to('cuda')

def save_checkpoint(state,filename='DenseNet.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def main():
    # Define your model, datasets, dataloaders, and training logic here
    EPOCHS = 50
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = sum(losses) / len(losses)
        scheduler.step()
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to('cuda'), val_labels.to('cuda')
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                val_losses.append(val_loss.item())

        val_avg_loss = sum(val_losses) / len(val_losses)
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Training Loss: {avg_loss:.4f}, Validation Loss: {val_avg_loss:.4f}')

        # Save the model if it has the lowest validation loss
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
            print(f'Model saved at epoch {epoch + 1} with Validation Loss: {val_avg_loss:.4f}')

    print('Training Done')

if __name__ == "__main__":
    main()
