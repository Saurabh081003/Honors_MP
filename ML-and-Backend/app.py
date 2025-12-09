"""
Kidney Stone Classification - Flask Backend
============================================
REST API for CT kidney image classification with explainability.
"""

import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# === CONFIGURATION ===
class Config:
    MODEL_PATH = "./datasets/checkpoints/best_hybrid_model.pth"
    IMG_SIZE = 224
    NUM_CLASSES = 4
    CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Class descriptions for explanations
    CLASS_INFO = {
        'Cyst': {
            'description': 'A fluid-filled sac that forms in or on the kidney.',
            'characteristics': [
                'Round or oval shape with smooth borders',
                'Fluid-filled appearance (darker region)',
                'Usually benign and asymptomatic',
                'Clear demarcation from surrounding tissue'
            ],
            'recommendation': 'Simple cysts typically require monitoring only. Complex cysts may need further evaluation.'
        },
        'Normal': {
            'description': 'Healthy kidney tissue with no abnormalities detected.',
            'characteristics': [
                'Uniform tissue density',
                'Normal kidney shape and size',
                'No masses or lesions visible',
                'Clear cortex-medulla differentiation'
            ],
            'recommendation': 'No immediate concerns. Continue regular health checkups.'
        },
        'Stone': {
            'description': 'Kidney stone (nephrolithiasis) - a hard deposit of minerals and salts.',
            'characteristics': [
                'High-density (bright) spot on CT',
                'Sharp or irregular edges',
                'Located in kidney, ureter, or bladder',
                'May cause obstruction or dilation'
            ],
            'recommendation': 'Treatment depends on size and location. Small stones may pass naturally; larger ones may require intervention.'
        },
        'Tumor': {
            'description': 'An abnormal growth of tissue in the kidney.',
            'characteristics': [
                'Irregular mass with unclear borders',
                'Variable density compared to normal tissue',
                'May show enhancement with contrast',
                'Potential distortion of kidney architecture'
            ],
            'recommendation': 'Further diagnostic workup recommended. Consult with a urologist or oncologist for proper evaluation.'
        }
    }


# === MODEL DEFINITION (same as training) ===
class HybridKidneyModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super(HybridKidneyModel, self).__init__()
        
        # Backbone networks
        self.resnet = models.resnet50(weights=None)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.densenet = models.densenet121(weights=None)
        self.densenet_features = self.densenet.features
        
        self.vgg = models.vgg16(weights=None)
        self.vgg_features = self.vgg.features
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(3584, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Store intermediate features for GradCAM
        self.gradients = None
        self.activations = None
    
    def save_gradient(self, grad):
        self.gradients = grad
    
    def forward(self, x, return_features=False):
        # ResNet features
        resnet_out = self.resnet_features(x)
        resnet_flat = resnet_out.view(resnet_out.size(0), -1)
        
        # DenseNet features
        densenet_out = self.densenet_features(x)
        
        # Register hook for GradCAM on densenet features
        if x.requires_grad:
            densenet_out.register_hook(self.save_gradient)
        self.activations = densenet_out
        
        densenet_flat = self.global_pool(densenet_out).view(densenet_out.size(0), -1)
        
        # VGG features
        vgg_out = self.vgg_features(x)
        vgg_flat = self.global_pool(vgg_out).view(vgg_out.size(0), -1)
        
        # Combine
        combined = torch.cat([resnet_flat, densenet_flat, vgg_flat], dim=1)
        output = self.classifier(combined)
        
        if return_features:
            return output, densenet_out
        return output


# === GRADCAM FOR EXPLAINABILITY ===
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def generate(self, input_tensor, target_class=None):
        """Generate GradCAM heatmap"""
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.model.gradients
        activations = self.model.activations
        
        if gradients is None or activations is None:
            return None
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(cam, size=(Config.IMG_SIZE, Config.IMG_SIZE), 
                           mode='bilinear', align_corners=False)
        
        return cam.squeeze().detach().cpu().numpy()


# === IMAGE PREPROCESSING ===
def get_transform():
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def apply_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    import cv2
    
    # Convert PIL to numpy
    img_array = np.array(image.resize((Config.IMG_SIZE, Config.IMG_SIZE)))
    
    # Normalize heatmap to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    
    blended = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(blended)


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# === FLASK APP ===
app = Flask(__name__, static_folder='static')
CORS(app)

# Global model and GradCAM
model = None
gradcam = None


def load_model():
    """Load the trained model"""
    global model, gradcam
    
    print(f"Loading model from {Config.MODEL_PATH}...")
    print(f"Using device: {Config.DEVICE}")
    
    model = HybridKidneyModel(num_classes=Config.NUM_CLASSES)
    
    if os.path.exists(Config.MODEL_PATH):
        checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully! (Val Acc: {checkpoint.get('val_acc', 'N/A')})")
    else:
        print(f"WARNING: Model file not found at {Config.MODEL_PATH}")
        print("Running with randomly initialized weights for testing.")
    
    model.to(Config.DEVICE)
    model.eval()
    
    gradcam = GradCAM(model)


@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('static', 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(Config.DEVICE),
        'classes': Config.CLASS_NAMES
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict kidney condition from CT image"""
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Load and preprocess image
        image = Image.open(file.stream).convert('RGB')
        original_image = image.copy()
        
        transform = get_transform()
        input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
        
        # Prediction
        with torch.set_grad_enabled(True):
            input_tensor.requires_grad_(True)
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = Config.CLASS_NAMES[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Get all class probabilities
            all_probs = {
                Config.CLASS_NAMES[i]: float(probabilities[0][i])
                for i in range(Config.NUM_CLASSES)
            }
            
            # Generate GradCAM heatmap
            try:
                heatmap = gradcam.generate(input_tensor, predicted_idx.item())
                if heatmap is not None:
                    heatmap_image = apply_heatmap(original_image, heatmap)
                    heatmap_base64 = image_to_base64(heatmap_image)
                else:
                    heatmap_base64 = None
            except Exception as e:
                print(f"GradCAM error: {e}")
                heatmap_base64 = None
        
        # Get class information for explanation
        class_info = Config.CLASS_INFO.get(predicted_class, {})
        
        # Build response
        response = {
            'success': True,
            'prediction': {
                'class': predicted_class,
                'confidence': round(confidence_score * 100, 2),
                'probabilities': {k: round(v * 100, 2) for k, v in all_probs.items()}
            },
            'explanation': {
                'description': class_info.get('description', ''),
                'characteristics': class_info.get('characteristics', []),
                'recommendation': class_info.get('recommendation', '')
            },
            'heatmap': heatmap_base64
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get information about all classes"""
    return jsonify({
        'classes': Config.CLASS_NAMES,
        'info': Config.CLASS_INFO
    })


# === MAIN ===
if __name__ == '__main__':
    load_model()
    print("\n" + "="*50)
    print("Kidney Stone Classification API")
    print("="*50)
    print(f"Server running at http://localhost:5000")
    print(f"API endpoints:")
    print(f"  GET  /api/health  - Health check")
    print(f"  POST /api/predict - Classify image")
    print(f"  GET  /api/classes - Get class info")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)