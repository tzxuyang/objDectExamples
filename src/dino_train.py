import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch  # torch
import utils
from PIL import Image
import logging
from sklearn.metrics import classification_report, accuracy_score
import os
import wandb

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_WANDB_KEY = "93205eda06a813b688c0462d11f09886a0cf7ae8"
_EPOCH = 240
_LR = 0.001
_BATCH_SIZE = 32

class DinoClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DinoClassifier, self).__init__()
        # Load the pre-trained DINOv3 model from timm
        self.backbone = timm.create_model('timm/vit_small_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0)
        # Freeze the backbone parameters
        self.backbone.eval()  # set the model in evaluation mode
        self.num_classes = num_classes
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Define a simple classification head
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output
    
    def get_train_transform(self):
        data_config = timm.data.resolve_model_data_config(self.backbone) 
        transform = timm.data.create_transform(
            **data_config,
            is_training=True
        )
        # print(data_config)
        # transform = transforms.Compose([
        #     transform_raw,
        #     transforms.RandomRotation(10),
        #     transforms.RandomHorizontalFlip(),
        #     # transforms.RandomCrop(224, padding=10),
        #     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # ])
        return transform
    
    def get_val_transform(self):
        data_config = timm.data.resolve_model_data_config(self.backbone) 
        transform = timm.data.create_transform(
            **data_config,
            is_training=False
        )
        return transform
    
    def get_info(self):
        backbone_size = sum(param.numel() for param in self.backbone.parameters())
        head_size = sum(param.numel() for param in self.head.parameters())
        total_size = backbone_size + head_size
        return self.backbone.num_features, self.num_classes, total_size
    
    def process_image(self, file_path):
        data_config = timm.data.resolve_model_data_config(self.backbone) 
        transforms = timm.data.create_transform(
            **data_config,
            is_training=False
        )
        if isinstance(file_path, str):
            image = Image.open(file_path).convert('RGB')
        else:
            image = file_path.convert('RGB')
        input_tensor = transforms(image).unsqueeze(0).to(device)
        return input_tensor
    
    def predict(self, input_tensor, class_names=None):
        with torch.no_grad():
            output = self.forward(input_tensor)
            prob = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(prob, 1)
        class_names = class_names
        # logging.info(f"Predicted: {class_names[predicted.item()]}, Confidence: {confidence.item():.4f}")
        return class_names[predicted.item()], confidence.item()
    
    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        logging.info(f"Model saved to {file_path}")

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        # Initialize the dataset with data source
        self.data = data
        self.labels = labels # Use torch.long for classification labels

    @classmethod
    def fromDirectory(cls, data_dir, label_dir, transform=None, scale = 1):
        # Load data and labels from a directory
        file_list = utils.create_file_list(data_dir)
        data = []
        labels = []
        for _ in range(scale):
            for file in file_list:
                # Load data and labels
                data.append(Image.open(file).convert('RGB'))
                label_file = file.replace(data_dir, label_dir).replace('.jpg', '.txt')
                with open(label_file, 'r') as f:
                    label = int(f.read().strip())
                    labels.append(label)
        if transform:
            data = [transform(img) for img in data]
        return cls(data, labels)

    def __len__(self):
        # Return the total number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieve a single sample at the given index
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def init_wandb(project_name="dino_classifier", wandb_key=_WANDB_KEY, config=None):
    wandb.login(key=wandb_key, relogin=True)
    if config is not None:
        config_in=config,
    else:
        config_in={
            "architecture": "DINOv3 + Custom Head",
            "dataset": "Port Classification",
            "epochs": _EPOCH,
            "batch_size": _BATCH_SIZE,
            "learning_rate": _EPOCH,
        }
    wandb.init(
        project=project_name,
        config=config_in
    )

def train_model(custom_model, train_loader, val_loader, **kwargs):
    custom_model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Adam to optimze the classification hear
    optimizer = optim.Adam(custom_model.head.parameters(), lr=kwargs.get('learning_rate', 0.001))
    num_epochs = kwargs.get('num_epochs', 150)
    for epoch in range(num_epochs):
        custom_model.train()  # set the model to train mode
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = custom_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss_train = running_loss / len(train_loader)

        custom_model.eval() # set the model to evaluation mode
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = custom_model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
        running_loss_val = running_loss / len(val_loader)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train loss: {running_loss_train:.4f}, Val loss: {running_loss_val:.4f}")
        wandb.log({
            "train_loss": running_loss_train,
            "val_loss": running_loss_val,
            "epoch": epoch
        })
    logging.info("Train completed")

def test_model(custom_model, test_loader, class_names):
    label_list = [i for i in range(len(class_names))]
    print(label_list)
    custom_model.to(device)
    custom_model.eval()  # set the model to evaluation mode
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = custom_model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    report = classification_report(all_labels, all_preds, labels = label_list, target_names=class_names)
    accuracy = accuracy_score(all_labels, all_preds)
    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:\n" + report)

if __name__ == "__main__":
    train_file_directory = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/train"
    train_label_directory = "/home/yang/MyRepos/tensorRT/datasets/port_cls/labels/train"
    test_file_directory = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/val"
    test_label_directory = "/home/yang/MyRepos/tensorRT/datasets/port_cls/labels/val"

    # init wandb
    init_wandb()

    # Load custom model
    custom_model = DinoClassifier(num_classes=6)
    dim, num_classes,size = custom_model.get_info()
    logging.info(f"Custom Dino Classifier model created. with vit dimension of {dim} and num_classes: {num_classes} and model size: {size/1e6}M parameters")
    logging.info(custom_model)

    # Prepare dataset and dataloader
    transforms = custom_model.get_train_transform()
    train_dataset = CustomDataset.fromDirectory(
        train_file_directory, 
        train_label_directory,
        transform= transforms
    )
    transforms = custom_model.get_val_transform()
    val_dataset = CustomDataset.fromDirectory(
        test_file_directory, 
        test_label_directory,
        transform = transforms
    )
    logging.info(f"train_dataset size : {int(train_dataset.__len__())}")
    sample, label = train_dataset.__getitem__(0)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=_BATCH_SIZE, 
        shuffle=True, 
        num_workers=12
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=12
    )
    # Train the model
    train_config = {
        'learning_rate': 0.001,
        'num_epochs': 240
    }
    train_model(custom_model, train_loader, val_loader, **train_config)

    # Save the trained model
    folders = os.listdir("./runs/cls")
    print(len(folders))
    if len(folders) > 1:
        folder_cnt = len(folders) -1
        print(folder_cnt)
        next_folder = f"train{folder_cnt}"
    else: #gitkeep
        next_folder = "train"
    os.makedirs(f"./runs/cls/{next_folder}/weights", exist_ok=True)
    path=f"./runs/cls/{next_folder}/weights/dino_classifier.pth"
    custom_model.save_model(path)
    logging.info(f"Model saved to {path}.")

    # Validate the model
    trained_model = DinoClassifier(num_classes=6)
    trained_model.load_state_dict(torch.load(path))
    trained_model.to(device)

    # inference on val dataset
    trained_model.eval()
    logging.info("--------------------------------------------------------------------------------------------------")
    val_dir = "/home/yang/MyRepos/tensorRT/datasets/port_cls/images/val"
    image_file_list = utils.create_file_list(val_dir)
    for image_path in image_file_list:
        image = Image.open(image_path).convert('RGB')
        # image.show()
        input_tensor = trained_model.process_image(image_path)
        class_name, confidence = trained_model.predict(input_tensor, class_names=['unplugged', 'port1', 'port2', 'port3', 'port4', 'port5'])
        logging.info(f"{image_path} classified as {class_name} with confidence {confidence:.4f}")

    # evaluate on test dataset
    logging.info("--------------------------------------------------------------------------------------------------")
    transforms = custom_model.get_val_transform()
    test_loader = DataLoader(
        val_dataset, 
        batch_size=len(val_dataset), 
        shuffle=False, 
        num_workers=12
    )
    test_model(trained_model, test_loader, class_names=['unplugged', 'port1', 'port2', 'port3', 'port4', 'port5'])