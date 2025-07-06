import os
import asyncio
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from gensim.models import Word2Vec
from datetime import datetime
from PIL import Image, ImageEnhance
import string
import re
from googlesearch import search
from playwright.async_api import async_playwright, Error
from pytesseract import pytesseract
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from . import models
from celery import shared_task
from celery_progress.backend import ProgressRecorder

# Configure pytesseract path
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define directories
OUTPUT_DIR = os.path.join("media/images")
ASSETS_DIR = os.path.join("media/assets/")

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Load Sastrawi stemmer and stopwords
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_factory = StopWordRemoverFactory()
sastrawi_stopwords = set(stopword_factory.get_stop_words())

class VisualModel(nn.Module):
    def __init__(self):
        super(VisualModel, self).__init__()
        # Load pretrained ResNet34
        from torchvision import models
        from torchvision.models import ResNet34_Weights
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

class SemanticModel(nn.Module):
    def __init__(self, embedding_dim=8, hidden_dim=8): 
        super(SemanticModel, self).__init__()
        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
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
            nn.Linear(2, 8),  # Increased capacity
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1),
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

# NEW MODEL LOADING FUNCTION
async def load_models(device):
    """Load all trained models"""
    try:
        # Model paths - UPDATE THESE PATHS
        visual_model_path = ASSETS_DIR + 'visual_model.pth'
        semantic_model_path = ASSETS_DIR + 'semantic_model.pth'
        combined_model_path = ASSETS_DIR + 'combined_model.pth'
        word2vec_path = ASSETS_DIR + 'word2vec_model.bin'
        
        # Load Word2Vec model
        word2vec_model = Word2Vec.load(word2vec_path)
        
        # Load Combined Model (which loads visual and semantic models internally)
        combined_model = CombinedModel(visual_model_path, semantic_model_path)
        combined_model.load_state_dict(torch.load(combined_model_path, map_location=device, weights_only=True))
        combined_model.to(device)
        combined_model.eval()
        
        print("All models loaded successfully")
        return combined_model, word2vec_model
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

# UPDATED PREDICTION FUNCTION
async def predict_single_website(model, word2vec_model, img_path, text, device='cpu'):
    """Make a prediction for a single website using the new model architecture"""
    
    try:
        print(f"Starting prediction for image: {img_path}")
        
        # Load and transform image
        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(img_path).convert('RGB')
        image = img_transform(image).unsqueeze(0).to(device)
        
        # Process text into word vectors
        words = text.split() if text else []
        word_vectors = []
        
        print(f"Processing {len(words)} words through Word2Vec")
        
        # Get word vectors from Word2Vec model
        for word in words:
            if word in word2vec_model.wv:
                word_vectors.append(word2vec_model.wv[word])
        
        # Handle empty vectors - IMPORTANT: Use correct vector size (200)
        if not word_vectors:
            word_vectors = np.zeros((1, word2vec_model.vector_size), dtype=np.float32)
        else:
            word_vectors = np.array(word_vectors, dtype=np.float32)
        
        # Convert to tensor
        word_vectors = torch.from_numpy(word_vectors).unsqueeze(0).to(device)
        text_length = torch.tensor([word_vectors.shape[1]], dtype=torch.long)
        
        # Make prediction using the combined model
        with torch.no_grad():
            outputs = model(image, word_vectors, text_length)
            
            final_pred = outputs['final_pred']
            visual_pred = outputs['visual_pred']
            semantic_pred = outputs['semantic_pred']
        
        # Convert to probabilities
        final_prob = final_pred.item()
        visual_prob = visual_pred.item()
        semantic_prob = semantic_pred.item()
        
        # Determine prediction class
        is_gambling = final_prob > 0.5        

        return {
            'text': text,
            'prediction': is_gambling,
            'visual_features': visual_prob,
            'semantic_features': semantic_prob,
            'combined_features': final_prob 
        }
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {"error": f"Prediction error: {str(e)}"}

# UPDATED PROCESS URL FUNCTION
async def process_url(url, model, word2vec_model, device):
    """Process a single URL and return prediction results"""
    try:
        # Take screenshot
        screenshot_path, filename = await take_screenshot(url)
        if screenshot_path is None:
            raise Exception(filename)
        
        # Perform OCR
        image = Image.open(screenshot_path)
        image = image.convert('L')  # Convert to grayscale
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)
        extracted_text = pytesseract.image_to_string(image, lang="ind")
        
        # Pre-process text
        filtered_text = preprocess_text(extracted_text)

        # Make prediction
        prediction = await predict_single_website(model, word2vec_model, screenshot_path, filtered_text, device)
        prediction['screenshot_path'] = filename
        prediction['url'] = url
        
        return prediction
    
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        return {"error": f"Processing error for {url}: {str(e)}", "url": url}

# UPDATED ANALYZE WEBSITES FUNCTION
async def analyze_websites(query=None, num_results=5, domain=None):
    """Main function to analyze websites for gambling content"""
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models using new function
    model, word2vec_model = await load_models(device)
    
    if not model or not word2vec_model:
        return {"error": "Failed to load models. Please check model paths and files."}
    
    if query:
        urls = get_search_results(query, num_results)
    else:
        urls = domain

    if not urls:
        return {"error": "No URLs found to analyze"}
    
    # Process all URLs concurrently
    tasks = [process_url(url, model, word2vec_model, device) for url in urls]
    results = await asyncio.gather(*tasks)
    
    return list(results)

# UPDATED CELERY TASKS
@shared_task(bind=True)
def image_upload(self, file_path):
    """Celery task to analyze uploaded images with progress tracking"""
    progress_recorder = ProgressRecorder(self)
    
    progress_recorder.set_progress(5, 100, "Loading analysis models...")
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run these synchronously in Celery task
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Load models using new function
    progress_recorder.set_progress(15, 100, "Loading gambling detection models...")
    model, word2vec_model = loop.run_until_complete(load_models(device))
    
    if not model or not word2vec_model:
        return {"error": "Failed to load models. Please check model paths and files."}
    
    progress_recorder.set_progress(35, 100, "Processing image...")
    
    # Perform OCR
    image = Image.open(file_path)
    image = image.convert('L')
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2) 
    
    progress_recorder.set_progress(50, 100, "Extracting text with OCR...")
    extracted_text = pytesseract.image_to_string(image, lang="ind")
    
    progress_recorder.set_progress(65, 100, "Analyzing text content...")
    
    # Pre-process text
    filtered_text = preprocess_text(extracted_text)
    
    progress_recorder.set_progress(80, 100, "Making prediction...")
    
    # Get filename from path
    filename = os.path.basename(file_path)
    
    # Make prediction
    prediction = loop.run_until_complete(predict_single_website(model, word2vec_model, file_path, filtered_text, device))
    prediction['screenshot_path'] = filename
    prediction['url'] = "image uploaded"
    
    progress_recorder.set_progress(90, 100, "Saving results...")
    
    # Save the result
    history_id = loop.run_until_complete(save_model([prediction], 'upload'))
    loop.close()
    
    progress_recorder.set_progress(100, 100, "Analysis complete!")
    return history_id

@shared_task(bind=True)
def main_domain(self, domain):
    """Celery task to analyze domain with progress tracking"""
    progress_recorder = ProgressRecorder(self)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Update progress
    progress_recorder.set_progress(10, 100, "Loading models...")
    
    # Run these synchronously in Celery task
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Load models using new function
    model, word2vec_model = loop.run_until_complete(load_models(device))
    
    if not model or not word2vec_model:
        return {"error": "Failed to load models. Please check model paths and files."}
    
    progress_recorder.set_progress(30, 100, f"Processing domain: {domain}")
    
    # If domain is a string, convert it to a list
    if isinstance(domain, str):
        domain = [domain]
    
    # Process domain
    results = loop.run_until_complete(analyze_websites(domain=domain))
    
    progress_recorder.set_progress(90, 100, "Saving results...")
    
    # Save results
    history_id = loop.run_until_complete(save_model(results, 'domain'))
    loop.close()
    
    progress_recorder.set_progress(100, 100, "Domain analysis complete!")
    return history_id

@shared_task(bind=True)
def main_search(self, query, num_results):
    progress_recorder = ProgressRecorder(self)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Update progress
    progress_recorder.set_progress(0, 100, "Loading models...")
    
    # Run these synchronously in Celery task
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Load models using new function
    model, word2vec_model = loop.run_until_complete(load_models(device))
    
    if not model or not word2vec_model:
        return {"error": "Failed to load models. Please check model paths and files."}
    
    # Get search results
    progress_recorder.set_progress(10, 100, f"Searching for: {query}")
    urls = get_search_results(query, num_results)
    if not urls:
        return {"error": "No URLs found to analyze"}
    
    # Process URLs one by one to show progress
    results = []
    total_urls = len(urls)
    for i, url in enumerate(urls):
        # Calculate percentage: 20% for model loading + search, 70% for processing URLs
        percentage = 20 + int((i / total_urls) * 70)
        progress_recorder.set_progress(percentage, 100, f"Processing {url}")
        
        # Process this URL
        result = loop.run_until_complete(process_url(url, model, word2vec_model, device))
        results.append(result)
    
    # Save results
    progress_recorder.set_progress(90, 100, "Saving results...")
    history_id = loop.run_until_complete(save_model(results, 'search'))
    loop.close()
    
    # Complete
    progress_recorder.set_progress(100, 100, "Processing complete!")

    return history_id

# Keep all other existing functions unchanged
def get_search_results(query: str, num_results: int = 5) -> list:
    """Get URLs from Google search results"""
    results = []
    try:
        for url in search(query, num_results=num_results):
            if url.startswith('/'):
                continue
            results.append(url)
    except Exception as e:
        print(f"Error during Google search: {e}")
    
    return results

async def take_screenshot(url: str, output_dir: str = OUTPUT_DIR) -> str:
    """Take a screenshot of a website"""
    # Make sure URL has a protocol
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Extract domain for filename
    domain = url.split('://')[1].split('/')[0].replace('.', '_')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{domain}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    async with async_playwright() as p:
        # Launch browser
        path_to_extension = ASSETS_DIR + 'cjpalhdlnbpafiamejdnhcphjbkeiagm-1.63.2-Crx4Chrome.com.crx'

        browser = await p.chromium.launch(args=[
        f"--no-sandbox",
        f"--ignore-certificate-errors",
        f"--disable-extensions-except={path_to_extension}",
        f"--load-extension={path_to_extension}"],
        headless=True, slow_mo=50)

        try:
            # Create a new page and navigate to URL
            context = await browser.new_context(viewport={"width": 1920, "height": 1080})
            page = await context.new_page()
            print(f"Navigating to {url}...")
            
            # Set a reasonable timeout (30 seconds)
            await page.goto(url, timeout=30000, wait_until="load")
            
            # Take screenshot
            await page.wait_for_timeout(1000)
            print(f"Taking screenshot...")
            await page.screenshot(path=filepath)
            print(f"Screenshot saved to: {filepath}")
            
            return filepath, filename
            
        except Error as e:
            print(f"Error accessing {url}: {e}")
            return None, e
        finally:
            await browser.close()

async def save_model(results, type):
    history_entry = await models.history.objects.acreate(type=type)
    for res in results:
        if 'error' in res:
            await models.result.objects.acreate(
                img=None,
                text=res['error'],
                visual_feature=None,
                semantic_feature=None,
                combined_feature=None,
                predict=None, 
                url=res['url'],
                history_id=history_entry
            )
        else:
            await models.result.objects.acreate(
                img=res['screenshot_path'],
                text=res['text'],
                visual_feature=res['visual_features'],
                semantic_feature=res['semantic_features'],
                combined_feature=res['combined_features'],
                predict=res['prediction'],
                url=res['url'],
                history_id=history_entry
            )
    return history_entry.id

def prepare_uploaded_image(file):
    """Helper function to save an uploaded file and return the path"""
    from django.core.files.storage import FileSystemStorage

    fs = FileSystemStorage(location='media/images/')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = fs.save('uploaded_'+timestamp+'.png', file)
    file_path = fs.path(filename)
    Image.open(file_path).save(file_path)
    
    return file_path

def preprocess_text(text):
    """Comprehensive text preprocessing for gambling detection"""
    try:
        text = text.lower()
    except (TypeError, AttributeError):
        text = str(text).lower()

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Remove punctuation but preserve spaces
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in sastrawi_stopwords and len(word) > 2]

    # Remove words with numbers
    filtered_words = [word for word in filtered_words if not any(char.isdigit() for char in word)]
    
    # Apply stemming
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # Join words with space for easier tokenization later
    return " ".join(stemmed_words)