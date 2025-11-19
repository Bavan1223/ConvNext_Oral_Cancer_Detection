import os
import torch
import timm
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms

# --- CONFIGURATION ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads' # Save inside static to display later
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- MODEL LOADING ---
CLASS_NAMES = ['non-cancerous', 'cancerous']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_trained_model():
    # Update this path if your model is in a different folder
    model_path = 'model/oral_cancer_model.pth' 
    try:
        model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=False, num_classes=len(CLASS_NAMES))
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
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
        # 1. Save the file to display it on the result page
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 2. Process and Predict
        try:
            img = Image.open(filepath).convert('RGB')
            input_tensor = inference_transforms(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            diagnosis = CLASS_NAMES[predicted_idx.item()].upper()
            conf_score = f"{confidence.item() * 100:.2f}%"

            # 3. Render the Result Page (Passing data to HTML)
            return render_template('result.html', 
                                   diagnosis=diagnosis, 
                                   confidence=conf_score, 
                                   image_url=filepath)
        except Exception as e:
            return f"Error analyzing image: {e}"

# Routes for other pages
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)