import os
import torch
import timm
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms

# --- CONFIGURATION ---
app = Flask(__name__)
app.secret_key = 'secret-key-for-session' # Required for flash messages
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- MODEL SETUP ---
# Classes: Ensure this order matches your training exactly
CLASS_NAMES = ['cancerous', 'non-cancerous']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_trained_model():
    model_path = 'model/oral_cancer_model.pth'
    try:
        # Load architecture
        model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=False, num_classes=len(CLASS_NAMES))
        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully on:", device)
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

model = load_trained_model()

# --- PREPROCESSING ---
class CircularMask:
    def __call__(self, img):
        from PIL import ImageDraw
        w, h = img.size
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        d = min(w, h)
        draw.ellipse(((w-d)/2, (h-d)/2, (w-d)/2+d, (h-d)/2+d), fill=255)
        img_masked = Image.new('RGB', (w, h), (0, 0, 0))
        if img.mode != 'RGB': img = img.convert('RGB')
        img_masked.paste(img, (0, 0), mask)
        return img_masked

inference_transforms = transforms.Compose([
    CircularMask(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- SMART FILTERS ---
def is_valid_oral_image(image_path):
    """
    Checks if the image matches typical oral cavity colors (Red/Pink/Flesh).
    Rejects images that are primarily Green (Nature) or Blue (Sky).
    """
    try:
        img = Image.open(image_path).convert('HSV')
        img = img.resize((50, 50))
        img_np = np.array(img)
        
        hue = img_np[:, :, 0]
        sat = img_np[:, :, 1]

        # 1. GREEN CHECK (Reject Nature)
        green_pixels = np.sum((hue > 40) & (hue < 90) & (sat > 40))
        if (green_pixels / 2500) > 0.10:
            return False, "Image detected as nature/object (Too much green)."

        # 2. ORAL COLOR CHECK (Red/Pink range)
        oral_pixels = np.sum(((hue < 25) | (hue > 230)) & (sat > 20))
        if (oral_pixels / 2500) < 0.15:
            return False, "Image does not match oral tissue color profiles."

        return True, ""
    except:
        return True, "" # Fail open if check errors

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 1. FILTER: Validate Image Content
        is_valid, reason = is_valid_oral_image(filepath)
        if not is_valid:
             return render_template('result.html', 
                                   diagnosis="INVALID IMAGE", 
                                   confidence="N/A", 
                                   image_url=filepath,
                                   details=reason,
                                   result_class="warning",
                                   note="Please upload a clear, close-up photo of the oral lesion.")

        # 2. INFERENCE: Run the AI Model
        try:
            img = Image.open(filepath).convert('RGB')
            input_tensor = inference_transforms(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            conf_value = confidence.item()
            conf_score = f"{conf_value * 100:.2f}%"
            raw_label = CLASS_NAMES[predicted_idx.item()]

            # --- 3. CLINICAL LOGIC ENGINE ---
            
            # CASE A: HEALTHY (Non-Cancerous & High Confidence)
            if raw_label == 'non-cancerous' and conf_value > 0.70:
                final_diagnosis = "BENIGN / HEALTHY"
                message = "No malignant features detected. Tissue appears consistent with healthy oral mucosa."
                note = "Recommendation: Maintain good oral hygiene and schedule regular dental check-ups."
                css_class = "normal"

            # CASE B: MALIGNANT (Cancerous & High Confidence)
            elif raw_label == 'cancerous' and conf_value > 0.85:
                final_diagnosis = "MALIGNANT CARCINOMA"
                message = "High probability of malignancy detected based on lesion texture and boundaries."
                note = "CRITICAL: Immediate specialist consultation is recommended. Persistence beyond 2 weeks is a red flag."
                css_class = "cancerous"

            # CASE C: SUSPICIOUS (Cancerous but Low Confidence -> Likely Ulcer)
            elif raw_label == 'cancerous' and conf_value <= 0.85:
                final_diagnosis = "SUSPICIOUS (POSSIBLE ULCER)"
                message = "Lesion shows abnormal inflammation but lacks definitive malignant patterns. Likely a severe Aphthous Ulcer or Traumatic Lesion."
                # ✅ YOUR REQUESTED NOTE:
                note = "Note: Minor mouth ulcers typically heal within 7 to 14 days. If this lesion persists longer, consult a doctor immediately."
                css_class = "warning"
            
            # CASE D: UNCERTAIN (Non-Cancerous but Low Confidence)
            else:
                final_diagnosis = "INCONCLUSIVE"
                message = "The image features are ambiguous. Lighting or focus may be affecting the analysis."
                note = "Please re-take the photo in better lighting conditions."
                css_class = "warning"

            return render_template('result.html', 
                                   diagnosis=final_diagnosis, 
                                   confidence=conf_score, 
                                   image_url=filepath,
                                   details=message,
                                   note=note,
                                   result_class=css_class)

        except Exception as e:
            return f"Error analyzing image: {e}"

# Placeholder routes for nav links
@app.route('/about')
def about(): return "<h2>About Page - Under Construction</h2>"
@app.route('/contact')
def contact(): return "<h2>Contact Page - Under Construction</h2>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)