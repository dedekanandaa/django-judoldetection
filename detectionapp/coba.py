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

# Model Architecture Definitions - must match the training code
class VisualFeatureExtractor(nn.Module):
    def __init__(self):
        super(VisualFeatureExtractor, self).__init__()
        # Use pretrained ResNet34
        from torchvision import models
        resnet = models.resnet34(pretrained=True)
        # Remove the classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Add a more complex feature processing network
        self.feature_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.feature_processor(x)
        return x

class SemanticFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(SemanticFeatureExtractor, self).__init__()
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )

        # Add a more complex feature processing network
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x, lengths):
        # Pack padded sequence for LSTM efficiency
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through LSTM
        packed_output, (hidden, _) = self.lstm(packed_input)

        # Concatenate the final hidden states from both directions
        hidden_forward = hidden[0, :, :]
        hidden_backward = hidden[1, :, :]
        hidden_cat = torch.cat((hidden_forward, hidden_backward), dim=1)

        # Process features
        output = self.feature_processor(hidden_cat)
        return output

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)

class LateFusionModel(nn.Module):
    def __init__(self, word_embedding_dim=100, lstm_hidden_dim=128, mlp_hidden_dim=128):
        super(LateFusionModel, self).__init__()
        # Feature extractors
        self.visual_extractor = VisualFeatureExtractor()
        self.semantic_extractor = SemanticFeatureExtractor(
            embedding_dim=word_embedding_dim,
            hidden_dim=lstm_hidden_dim
        )

        # Single-modal classifiers
        self.visual_classifier = Classifier(256, mlp_hidden_dim)
        self.semantic_classifier = Classifier(256, mlp_hidden_dim)

        # Combined features classifier
        self.combined_classifier = Classifier(512, mlp_hidden_dim * 2)

        # Late fusion layer (weighted fusion)
        self.fusion_layer = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize fusion layer with bias toward combined prediction
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, image, text, text_lengths):
        # Extract features
        visual_features = self.visual_extractor(image)
        semantic_features = self.semantic_extractor(text, text_lengths)

        # Combined features
        combined_features = torch.cat((visual_features, semantic_features), dim=1)

        # Single-modal classifications
        visual_pred = self.visual_classifier(visual_features)
        semantic_pred = self.semantic_classifier(semantic_features)
        combined_pred = self.combined_classifier(combined_features)

        # Late fusion
        fusion_input = torch.cat((visual_pred, semantic_pred, combined_pred), dim=1)
        final_output = self.fusion_layer(fusion_input)

        return {
            'visual_pred': visual_pred,
            'semantic_pred': semantic_pred,
            'combined_pred': combined_pred,
            'final_pred': final_output,
            'visual_features': visual_features,
            'semantic_features': semantic_features
        }


async def load_model(model_path, device):
    """Load the trained model"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Define default model parameters matching training constants
        WORD_EMBEDDING_DIM = 100
        LSTM_HIDDEN_DIM = 128
        MLP_HIDDEN_DIM = 128
        
        # Check if it's the production model package or training checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # It's a training checkpoint
            try:
                params = torch.load(os.path.join(ASSETS_DIR, 'model_params.pth'))
                model = LateFusionModel(
                    word_embedding_dim=params.get('word_embedding_dim', WORD_EMBEDDING_DIM),
                    lstm_hidden_dim=params.get('lstm_hidden_dim', LSTM_HIDDEN_DIM),
                    mlp_hidden_dim=params.get('mlp_hidden_dim', MLP_HIDDEN_DIM)
                ).to(device)
            except FileNotFoundError:
                # Fallback to default parameters if params file not found
                print("Model parameters file not found, using default values")
                model = LateFusionModel(
                    word_embedding_dim=WORD_EMBEDDING_DIM,
                    lstm_hidden_dim=LSTM_HIDDEN_DIM,
                    mlp_hidden_dim=MLP_HIDDEN_DIM
                ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            # It's a production model package
            model = checkpoint['model'].to(device)
        else:
            # It's a direct model
            model = checkpoint.to(device)
            
        model.eval()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


async def load_word2vec_model(word2vec_path):
    """Load the Word2Vec model"""
    try:
        word2vec_model = Word2Vec.load(word2vec_path)
        print("Word2Vec model loaded successfully")
        return word2vec_model
    except Exception as e:
        print(f"Error loading Word2Vec model: {e}")
        return None


async def predict_single_website(model, word2vec_model, img_path, text, device='cpu'):
    """Make a prediction for a single website"""
    
    try:
        # Add debugging 
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
        
        # Debug text processing
        print(f"Processing {len(words)} words through Word2Vec")
        
        # Track how many words were found in the model
        for word in words:
            if word in word2vec_model.wv:
                word_vectors.append(word2vec_model.wv[word])
        
        # If no words found in Word2Vec model, use zero vector
        if not word_vectors:
            word_vectors.append(np.zeros(word2vec_model.vector_size))
        
        word_vectors = np.array(word_vectors)
        word_vectors = torch.FloatTensor(word_vectors).unsqueeze(0).to(device)
        text_length = torch.tensor([len(word_vectors[0])])
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image, word_vectors, text_length)
            final_pred = outputs['final_pred']
            visual_pred = outputs['visual_pred']
            semantic_pred = outputs['semantic_pred']
            combined_pred = outputs['combined_pred']
        
        # Convert to probabilities
        final_prob = final_pred.item()
        visual_prob = visual_pred.item()
        semantic_prob = semantic_pred.item()
        combined_prob = combined_pred.item()
        
        # Determine prediction class
        is_gambling = final_prob > 0.5
        is_gambling = True if is_gambling else False
        
        components = [
            ('visual', visual_prob),
            ('semantic', semantic_prob),
            ('combined', combined_prob)
        ]

        if is_gambling:
            max_component = max(components, key=lambda x: x[1])  # Get highest probability component
        else:
            max_component = min(components, key=lambda x: x[1])  # Get lowest probability component
        
        return {
            'prediction': is_gambling,
            'confidence': final_prob,
            'component_contributions': {
                'visual_features': visual_prob,
                'semantic_features': semantic_prob,
                'combined_features': combined_prob
            },
            'primary_factor': max_component[0],
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {"error": f"Prediction error: {str(e)}"}


def get_search_results(query: str, num_results: int = 5) -> list:
    """Get URLs from Google search results"""
    results = []
    try:
        for url in search(query, num_results=num_results):
            if url.startswith('/'):
                # Skip relative URLs
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
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            
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


async def process_url(url, model, word2vec_model, device):
    """Process a single URL and return prediction results"""
    try:
        # Take screenshot
        screenshot_path, filename = await take_screenshot(url)
        if screenshot_path is None:
            return {"error": filename}
        
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
        prediction['screenshot_path'] = filename  # Add the screenshot path
        prediction['extracted_text'] = filtered_text  # Add a preview of the text
        prediction['url'] = url  # Add the URL to the prediction result
        
        return prediction
    
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        return {"error": f"Processing error for {url}: {str(e)}", "url": url}


async def analyze_websites(query=None, num_results=5, domain=None):
    """Main function to analyze websites for gambling content"""
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    model_path = ASSETS_DIR + 'best_gambling_detector_model.pth'
    word2vec_path = ASSETS_DIR + 'word2vec_model.bin'
    
    model = await load_model(model_path, device)
    word2vec_model = await load_word2vec_model(word2vec_path)
    
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
    
    # Always return a list of results, even if only one
    return list(results)

async def save_model(results, type):
    history_entry = await models.history.objects.acreate(type=type)
    for res in results:
        if 'error' in res:
            await models.result.objects.acreate(
                img=None,
                text=res['error'],
                primary_factor=None,
                visual_feature=None,
                semantic_feature=None,
                combined_feature=None,
                confidence=None,
                predict=None, 
                url=res['url'],
                history_id=history_entry
            )
        else:
            await models.result.objects.acreate(
                img=res['screenshot_path'],
                text=res['extracted_text'],
                primary_factor=res['primary_factor'],
                visual_feature=res['component_contributions']['visual_features'],
                semantic_feature=res['component_contributions']['semantic_features'],
                combined_feature=res['component_contributions']['combined_features'],
                confidence=res['confidence'],
                predict=res['prediction'],
                url=res['url'],
                history_id=history_entry
            )
    return history_entry.id  # Return the history entry ID


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

@shared_task(bind=True)
def image_upload(self, file_path):
    """Celery task to analyze uploaded images with progress tracking"""
    progress_recorder = ProgressRecorder(self)
    
    progress_recorder.set_progress(5, 100, "Loading analysis models...")
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    model_path = ASSETS_DIR + 'best_gambling_detector_model.pth'
    word2vec_path = ASSETS_DIR + 'word2vec_model.bin'
    
    # Run these synchronously in Celery task
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Show progress for each model
    progress_recorder.set_progress(15, 100, "Loading gambling detection model...")
    model = loop.run_until_complete(load_model(model_path, device))
    
    progress_recorder.set_progress(25, 100, "Loading language model...")
    word2vec_model = loop.run_until_complete(load_word2vec_model(word2vec_path))
    
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
    prediction['extracted_text'] = filtered_text
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
    
    # Load models
    model_path = ASSETS_DIR + 'best_gambling_detector_model.pth'
    word2vec_path = ASSETS_DIR + 'word2vec_model.bin'
    
    # Run these synchronously in Celery task
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    model = loop.run_until_complete(load_model(model_path, device))
    word2vec_model = loop.run_until_complete(load_word2vec_model(word2vec_path))
    
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
    
    # Load models
    model_path = ASSETS_DIR + 'best_gambling_detector_model.pth'
    word2vec_path = ASSETS_DIR + 'word2vec_model.bin'
    
    # Update progress
    progress_recorder.set_progress(0, 100, "Loading models...")
    
    # Run these synchronously in Celery task
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    model = loop.run_until_complete(load_model(model_path, device))
    word2vec_model = loop.run_until_complete(load_word2vec_model(word2vec_path))
    
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

    # Return the history_id which will be passed to onProgressSuccess in progress.html
    return history_id


# For standalone testing
if __name__ == "__main__":
    query = "berita"
    num_results = 5
    results = asyncio.run(analyze_websites(query=query, num_results=num_results))
    
    # Print summary
    print("\n===== ANALYSIS SUMMARY =====")
    for key, value in results["summary"].items():
        print(f"{key}: {value}")
    
    # Print gambling sites
    print("\n===== GAMBLING SITES =====")
    for site in results["gambling_sites"]:
        print(f"URL: {site['url']}")
        print(f"Confidence: {site['confidence']:.2f}")
        print(f"Primary factor: {site['primary_factor']}")
        print("---")
    
    # Print errors
    if results["errors"]:
        print("\n===== ERRORS =====")
        for error in results["errors"]:
            print(f"URL: {error.get('url', 'Unknown')}")
            print(f"Error: {error.get('error', 'Unknown error')}")
            print("---")