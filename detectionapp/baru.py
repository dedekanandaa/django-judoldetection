import os
import torch
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet34_Weights


class VisualDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.img_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

class VisualModel(nn.Module):
    def __init__(self):
        super(VisualModel, self).__init__()

        # Load pretrained ResNet34
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

class SemanticDataset(Dataset):
    def __init__(self, txt_paths, labels, word2vec_model):
        self.txt_paths = txt_paths
        self.labels = labels
        self.word2vec_model = word2vec_model

    def __len__(self):
        return len(self.txt_paths)

    def __getitem__(self, idx):
        # Load text
        with open(self.txt_paths[idx], 'r', encoding='utf-8') as f:
            text = f.read()

        # Preprocess text
        tokens = re.findall(r'\b\w+\b', text)

        # Convert to vectors - PERBAIKAN DI SINI
        word_vectors = []
        for word in tokens:
            if word in self.word2vec_model.wv:
                word_vectors.append(self.word2vec_model.wv[word])

        # Handle empty vectors
        if not word_vectors:
            word_vectors = np.zeros((1, 100), dtype=np.float32)
        else:
            # Konversi ke numpy array dulu, baru ke tensor
            word_vectors = np.array(word_vectors, dtype=np.float32)

        # Konversi ke tensor
        word_vectors = torch.from_numpy(word_vectors)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return word_vectors, label

class SemanticModel(nn.Module):
    def __init__(self, embedding_dim=100, hidden_dim=128):
        super(SemanticModel, self).__init__()

        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths):
        # Pack sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_output, (hidden, _) = self.lstm(packed_input)

        # Combine bidirectional hidden states
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_combined = torch.cat((hidden_forward, hidden_backward), dim=1)

        # Classify
        output = self.classifier(hidden_combined)
        return output

class CombinedDataset(Dataset):
    def __init__(self, img_paths, txt_paths, labels, word2vec_model, img_transform=None):
        self.img_paths = img_paths
        self.txt_paths = txt_paths
        self.labels = labels
        self.word2vec_model = word2vec_model
        self.img_transform = img_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.img_paths[idx]).convert('RGB')
        if self.img_transform:
            image = self.img_transform(image)

        # Load text
        with open(self.txt_paths[idx], 'r', encoding='utf-8') as f:
            text = f.read()

        # Preprocess text - CONSISTENT WITH SEMANTIC MODEL
        tokens = re.findall(r'\b\w+\b', text.lower())

        # Convert to vectors - FIXED CONSISTENCY
        word_vectors = []
        for word in tokens:
            if word in self.word2vec_model.wv:
                word_vectors.append(self.word2vec_model.wv[word])

        # Handle empty vectors - CONSISTENT SHAPE
        if not word_vectors:
            word_vectors = np.zeros((1, 100), dtype=np.float32)  # Shape (1, 100)
        else:
            word_vectors = np.array(word_vectors, dtype=np.float32)

        word_vectors = torch.from_numpy(word_vectors)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, word_vectors, label

class CombinedModel(nn.Module):
    def __init__(self, visual_model_path, semantic_model_path):
        super(CombinedModel, self).__init__()

        # Load pre-trained models
        self.visual_model = VisualModel()
        self.visual_model.load_state_dict(torch.load(visual_model_path, map_location='cpu', weights_only=True))

        self.semantic_model = SemanticModel()
        self.semantic_model.load_state_dict(torch.load(semantic_model_path, map_location='cpu', weights_only=True))

        # Option to freeze pre-trained models (recommended for stability)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        # Improved fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(2, 32),  # Increased capacity
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text, text_lengths):
        # Get predictions from individual models
        with torch.no_grad():  # Since we froze the models
            visual_pred = self.visual_model(image)
            semantic_pred = self.semantic_model(text, text_lengths)

        # Combine predictions
        combined_input = torch.cat([visual_pred, semantic_pred], dim=1)
        final_pred = self.fusion_layer(combined_input)

        return {
            'visual_pred': visual_pred,
            'semantic_pred': semantic_pred,
            'final_pred': final_pred
        }


def collate_semantic_fn(batch):
    """Custom collate function untuk semantic data"""
    # Sort by sequence length (descending)
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

    texts = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])

    # Get sequence lengths
    text_lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)

    # Pad sequences
    max_len = max(text_lengths).item()
    batch_size = len(texts)
    embedding_dim = texts[0].shape[1]

    padded_texts = torch.zeros(batch_size, max_len, embedding_dim, dtype=torch.float32)

    for i, text in enumerate(texts):
        seq_len = min(text.shape[0], max_len)
        padded_texts[i, :seq_len, :] = text[:seq_len]

    return padded_texts, text_lengths, labels

def collate_combined_fn(batch):
    # Sort by text length
    batch = sorted(batch, key=lambda x: x[1].shape[0], reverse=True)

    images = torch.stack([item[0] for item in batch])
    texts = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])

    # Get text lengths - FIXED TYPE
    text_lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)

    # Pad texts - IMPROVED EFFICIENCY
    max_len = max(text_lengths).item()
    batch_size = len(texts)
    embedding_dim = texts[0].shape[1]

    # Create padded tensor directly
    padded_texts = torch.zeros(batch_size, max_len, embedding_dim, dtype=torch.float32)

    # Fill with data
    for i, text in enumerate(texts):
        seq_len = min(text.shape[0], max_len)  # Prevent overflow
        padded_texts[i, :seq_len, :] = text[:seq_len]

    return images, padded_texts, text_lengths, labels

def prepare_semantic_data(data_dir):
    """Siapkan data untuk training semantic model"""
    txt_paths = []
    labels = []

    # Gambling texts
    gambling_txt_dir = os.path.join(data_dir, 'gambling', 'texts')
    if os.path.exists(gambling_txt_dir):
        for filename in os.listdir(gambling_txt_dir):
            if filename.endswith('.txt'):
                txt_path = os.path.join(gambling_txt_dir, filename)
                if os.path.getsize(txt_path) > 0:  # Validate file size
                    txt_paths.append(txt_path)
                    labels.append(1)

    # Non-gambling texts
    non_gambling_txt_dir = os.path.join(data_dir, 'non_gambling', 'texts')
    if os.path.exists(non_gambling_txt_dir):
        for filename in os.listdir(non_gambling_txt_dir):
            if filename.endswith('.txt'):
                txt_path = os.path.join(non_gambling_txt_dir, filename)
                if os.path.getsize(txt_path) > 0:  # Validate file size
                    txt_paths.append(txt_path)
                    labels.append(0)

    print(f"Found {len(txt_paths)} text files")
    print(f"Gambling: {labels.count(1)}, Non-gambling: {labels.count(0)}")

    return txt_paths, labels

def prepare_visual_data(data_dir):
    """Siapkan data untuk training visual model"""
    img_paths = []
    labels = []

    # Gambling images
    gambling_img_dir = os.path.join(data_dir, 'gambling', 'images')
    for filename in os.listdir(gambling_img_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_paths.append(os.path.join(gambling_img_dir, filename))
            labels.append(1)

    # Non-gambling images
    non_gambling_img_dir = os.path.join(data_dir, 'non_gambling', 'images')
    for filename in os.listdir(non_gambling_img_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_paths.append(os.path.join(non_gambling_img_dir, filename))
            labels.append(0)

    return img_paths, labels

def prepare_combined_data(data_dir):
    """Siapkan data untuk training combined model dengan validasi file"""
    img_paths = []
    txt_paths = []
    labels = []

    # Gambling data
    gambling_img_dir = os.path.join(data_dir, 'gambling', 'images')
    gambling_txt_dir = os.path.join(data_dir, 'gambling', 'texts')

    if os.path.exists(gambling_img_dir) and os.path.exists(gambling_txt_dir):
        for filename in os.listdir(gambling_img_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(gambling_img_dir, filename)
                txt_filename = filename.rsplit('.', 1)[0] + '.txt'
                txt_path = os.path.join(gambling_txt_dir, txt_filename)

                if os.path.exists(txt_path):
                        img_paths.append(img_path)
                        txt_paths.append(txt_path)
                        labels.append(1)

    # Non-gambling data
    non_gambling_img_dir = os.path.join(data_dir, 'non_gambling', 'images')
    non_gambling_txt_dir = os.path.join(data_dir, 'non_gambling', 'texts')

    if os.path.exists(non_gambling_img_dir) and os.path.exists(non_gambling_txt_dir):
        for filename in os.listdir(non_gambling_img_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(non_gambling_img_dir, filename)
                txt_filename = filename.rsplit('.', 1)[0] + '.txt'
                txt_path = os.path.join(non_gambling_txt_dir, txt_filename)

                if os.path.exists(txt_path):
                        img_paths.append(img_path)
                        txt_paths.append(txt_path)
                        labels.append(0)

    print(f"Found {len(img_paths)} valid image-text pairs")
    print(f"Gambling: {labels.count(1)}, Non-gambling: {labels.count(0)}")

    return img_paths, txt_paths, labels

def load_model(model_type, model_path, visual_model_path=None, semantic_model_path=None):
    """Load model berdasarkan tipe"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "visual":
        model = VisualModel()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    elif model_type == "semantic":
        model = SemanticModel()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    elif model_type == "combined":
        model = CombinedModel(visual_model_path, semantic_model_path)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        raise ValueError("Model type must be 'visual', 'semantic', or 'combined'")
    
    model.to(device)
    model.eval()
    return model, device

def test_visual_model(data_dir, model_path):
    """Test visual model"""
    print("Testing Visual Model...")
    print("=" * 50)
    
    # Load model
    model, device = load_model("visual", model_path)
    
    # Prepare test data
    img_paths, labels = prepare_visual_data(data_dir)
    
    # Split data (gunakan test set yang sama)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        img_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Image transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dataset = VisualDataset(X_test, y_test, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test model
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            probs = outputs.squeeze().cpu().numpy()
            preds = (outputs.squeeze() > 0.5).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    return all_labels, all_preds, all_probs

def test_semantic_model(data_dir, model_path, word2vec_path):
    """Test semantic model"""
    print("Testing Semantic Model...")
    print("=" * 50)
    
    # Load models
    model, device = load_model("semantic", model_path)
    word2vec_model = Word2Vec.load(word2vec_path)
    
    # Prepare test data
    txt_paths, labels = prepare_semantic_data(data_dir)
    
    # Split data (gunakan test set yang sama)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        txt_paths, labels, test_size=0.2, random_state=42,
        stratify=labels if len(set(labels)) > 1 else None
    )
    
    # Create test dataset
    test_dataset = SemanticDataset(X_test, y_test, word2vec_model)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                            collate_fn=collate_semantic_fn, num_workers=0)
    
    # Test model
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for texts, text_lengths, labels in test_loader:
            texts, text_lengths = texts.to(device), text_lengths.to(device)
            outputs = model(texts, text_lengths)
            
            probs = outputs.squeeze().cpu().numpy()
            preds = (outputs.squeeze() > 0.5).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    return all_labels, all_preds, all_probs

def test_combined_model(data_dir, model_path, visual_model_path, semantic_model_path, word2vec_path):
    """Test combined model"""
    print("Testing Combined Model...")
    print("=" * 50)
    
    # Load models
    model, device = load_model("combined", model_path, visual_model_path, semantic_model_path)
    word2vec_model = Word2Vec.load(word2vec_path)
    
    # Prepare test data
    img_paths, txt_paths, labels = prepare_combined_data(data_dir)
    
    # Split data (gunakan test set yang sama)
    from sklearn.model_selection import train_test_split
    _, X_test_img, _, X_test_txt, _, y_test = train_test_split(
        img_paths, txt_paths, labels, test_size=0.2, random_state=42,
        stratify=labels if len(set(labels)) > 1 else None
    )
    
    # Image transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dataset = CombinedDataset(X_test_img, X_test_txt, y_test, word2vec_model, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                            collate_fn=collate_combined_fn, num_workers=0)
    
    # Test model
    all_preds, all_labels, all_probs = [], [], []
    visual_preds, semantic_preds = [], []
    
    with torch.no_grad():
        for images, texts, text_lengths, labels in test_loader:
            images, texts = images.to(device), texts.to(device)
            outputs = model(images, texts, text_lengths)
            
            # Final predictions
            probs = outputs['final_pred'].squeeze().cpu().numpy()
            preds = (outputs['final_pred'].squeeze() > 0.5).cpu().numpy()
            
            # Individual model predictions
            visual_pred = (outputs['visual_pred'].squeeze() > 0.5).cpu().numpy()
            semantic_pred = (outputs['semantic_pred'].squeeze() > 0.5).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            visual_preds.extend(visual_pred)
            semantic_preds.extend(semantic_pred)
    
    return all_labels, all_preds, all_probs, visual_preds, semantic_preds
