from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___healthy",
    "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca",
    "Grape___Leaf_blight",
    "Grape___healthy",
    "Orange___Huanglongbing",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper_bell___Bacterial_spot",
    "Pepper_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Tomato_yellow_leaf_curl_virus",
    "Tomato___healthy",
]

DISEASES_DB = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple scab",
        "description": "A fungal disease causing olive-green to brown velvety spots on leaves and fruit.",
        "symptoms": ["Olive-green to brown velvety spots on leaves", "Spots may become raised and cracked", "Fruit may develop severe scabbing"],
        "treatment": ["Remove and destroy infected leaves and fruit", "Apply fungicide containing captan or myclobutanil", "Prune for better air circulation"],
        "prevention": ["Rake and dispose of fallen leaves", "Apply dormant-season fungicide", "Plant resistant varieties"],
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black rot",
        "description": "A fungal disease causing dark, sunken lesions on fruits and cankers on branches.",
        "symptoms": ["Small, firm, light-brown spots on fruit that enlarge and turn black", "Sunken lesions with concentric rings", "Cankers on twigs and branches"],
        "treatment": ["Prune out infected canes and mummified fruit", "Apply fungicide during growing season", "Remove and destroy affected fruit"],
        "prevention": ["Remove all mummified fruit from trees and ground", "Prune for open canopy", "Apply preventive fungicide sprays"],
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar apple rust",
        "description": "A fungal disease requiring cedar trees as alternate hosts, causing orange rust spots on leaves and fruit.",
        "symptoms": ["Bright orange-yellow spots on upper leaf surface", "Spots turn black in later seasons", "Orange protrusions on underside of leaves"],
        "treatment": ["Apply fungicide containing sulfur or myclobutanil", "Remove nearby cedar host plants", "Collect and destroy fallen leaves"],
        "prevention": ["Eliminate cedar trees within 500 feet", "Plant resistant apple varieties", "Apply preventive sprays at pink bud stage"],
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "description": " healthy apple tree with no signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Maintain proper orchard hygiene", "Regular monitoring for early signs of disease", "Ensure adequate water and nutrients"],
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "description": " healthy blueberry bush with no signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Maintain proper soil pH (4.5-5.5)", "Regular monitoring for pests and diseases", "Proper irrigation management"],
    },
    "Cherry___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "description": " healthy cherry tree with no signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Prune for air circulation", "Remove fallen leaves and debris", "Monitor for bacterial infections"],
    },
    "Cherry___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery mildew",
        "description": "A fungal disease causing white powdery growth on leaves and fruits.",
        "symptoms": ["White powdery growth on leaves, stems, and fruit", "Leaves may become distorted or curled", "Fruit may have reduced quality"],
        "treatment": ["Apply fungicide containing sulfur or potassium bicarbonate", "Remove affected leaves", "Improve air circulation around plants"],
        "prevention": ["Plant in full sun with good air circulation", "Avoid overhead watering", "Apply preventive fungicide sprays in spring"],
    },
    "Corn___Cercospora_leaf_spot": {
        "plant": "Corn",
        "disease": "Cercospora leaf spot",
        "description": "A fungal disease causing grayish-brown lesions on corn leaves with purple borders.",
        "symptoms": ["Grayish-brown lesions with purple borders on leaves", "Lesions may merge causing large dead areas", "Premature leaf death"],
        "treatment": ["Apply fungicide containing azoxystrobin or chlorothalonil", "Remove severely infected leaves", "Crop rotation with non-host plants"],
        "prevention": ["Rotate crops annually", "Residue management after harvest", "Plant resistant hybrids"],
    },
    "Corn___Common_rust": {
        "plant": "Corn",
        "disease": "Common rust",
        "description": "A fungal disease causing reddish-brown pustules on corn leaves.",
        "symptoms": ["Small, reddish-brown pustules on leaf surfaces", "Pustules release orange-brown spores", "Leaves may prematurely die"],
        "treatment": ["Apply fungicide containing triazoles or strobilurins", "Remove infected plant material", "Early planting to avoid peak infection"],
        "prevention": ["Plant rust-resistant hybrids", "Rotate crops", "Destroy crop residues"],
    },
    "Corn___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "description": "A fungal disease causing large, grayish-green to tan lesions on corn leaves.",
        "symptoms": ["Large, dull gray-green to tan rectangular lesions", "Lesions may extend along the leaf midrib", "Premature leaf death from bottom up"],
        "treatment": ["Apply fungicide containing azoxystroin or pyraclostrobin", "Remove infected residue", "Crop rotation"],
        "prevention": ["Rotate crops", "Plow under residue", "Plant resistant hybrids"],
    },
    "Corn___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "description": " healthy corn plant with no signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Maintain field hygiene", "Rotate crops", "Manage nitrogen levels properly"],
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black rot",
        "description": "A fungal disease causing small black spots on leaves and fruit with circular lesions.",
        "symptoms": ["Small purple spots on leaves that enlarge with black center and yellow halo", "Black, sunken spots on fruit", "Fruit shrivels and mummifies"],
        "treatment": ["Apply fungicide containing captan or sulfur", "Remove mummified fruit from vines", "Prange for better air circulation"],
        "prevention": ["Early-season fungicide sprays", "Remove and destroy mummified fruit", "Prune for open canopy"],
    },
    "Grape___Esca": {
        "plant": "Grape",
        "disease": "Esca",
        "description": "A degenerative fungal disease causing leaf necrosis and wood decay in grape vines.",
        "symptoms": ["Interleafal chlorosis and necrosis (black measles)", "Streaking or piebbling of wood tissue", "Dieback of shoots and canes"],
        "treatment": ["Prune out infected canes", "Apply fungicides to wounds", "Maintain vine vigor"],
        "prevention": ["Avoid unnecessary pruning during wet weather", "Remove and destroy infected wood", "Health vineyard management"],
    },
    "Grape___Leaf_blight": {
        "plant": "Grape",
        "disease": "Leaf blight",
        "description": "A fungal disease causing leaf spots and blighting in grape vines.",
        "symptoms": ["Rectangular dark spots on leaves", "Spots may coalesce causing large blighted areas", "Premature leaf drop"],
        "treatment": ["Apply fungicide containing chlorothalonil or copper", "Remove affected leaves", "Improve vineyard airflow"],
        "prevention": ["Remove infected plant material", "Maintain proper vine spacing", "Apply preventive sprays during wet seasons"],
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "description": " healthy grape vine with no signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Regular monitoring", "Proper canopy management", "Balanced nutrition"],
    },
    "Orange___Huanglongbing": {
        "plant": "Orange",
        "disease": "Huanglongbing (Citrus greening)",
        "description": "A bacterial disease causing yellowing of leaves and deformed fruit, ultimately killing the tree.",
        "symptoms": ["Yellowing of veins and blotchy mottling of leaves", "Small, lopsided, poorly colored fruit", "Degraded fruit taste"],
        "treatment": ["No cure available", "Remove and infected trees", "Control psyllid vector with insecticides"],
        "prevention": ["Monitor and control Asian citrus psyllid", "Remove infected trees promptly", "Use disease-free planting material"],
    },
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial spot",
        "description": "A bacterial disease causing small, dark spots on leaves and fruit with yellow halos.",
        "symptoms": ["Small, water-soaked spots on leaves with yellow halos", "Lesions on fruit becoming raised and scabby", "Defoliation in severe cases"],
        "treatment": ["Apply copper-based bactericides", "Remove and destroy infected plant material", "Improve air circulation"],
        "prevention": ["Apply dormant copper sprays", "Avoid working with wet plants", "Plant resistant varieties"],
    },
    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "description": " healthy peach tree with no signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Regular monitoring", "Proper pruning for air circulation", "Maintain tree vigor"],
    },
    "Pepper_bell___Bacterial_spot": {
        "plant": "Pepper",
        "disease": "Bacterial spot",
        "description": "A bacterial disease causing small, dark spots on pepper leaves and fruit with halos.",
        "symptoms": ["Small, dark brown spots on leaves with yellow halos", "Spots on fruit becoming raised and scabby", "Premature leaf drop"],
        "treatment": ["Apply copper-based bactericides", "Remove affected leaves", "Rotate crops"],
        "prevention": ["Avoid overhead watering", "Remove plant debris", "Rotate with non-host crops"],
    },
    "Pepper_bell___healthy": {
        "plant": "Pepper",
        "disease": "Healthy",
        "description": " healthy pepper plant with no signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Regular monitoring", "Proper spacing for air circulation", "Remove plant debris"],
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early blight",
        "description": "A fungal disease causing dark concentric spots on potato leaves and stems.",
        "symptoms": ["Dark brown concentric spots on lower leaves", "Lesions may have target-like rings", "Premature leaf death"],
        "treatment": ["Apply fungicide containing chlorothalonil or mancozeb", "Remove infected leaves", "Ensure proper spacing"],
        "prevention": ["Rotate crops", "Mulch to reduce soil splash", "Water at base, not foliage"],
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late blight",
        "description": "A highly destructive fungal disease causing rapid blighting of leaves and fruit.",
        "symptoms": ["Water-soaked lesions on leaves that rapidly expand", "White fuzzy growth on underside of leaves in humid conditions", "Dark, firm spots on tubers"],
        "treatment": ["Apply fungicide containing chlorothalonil or mancozeb immediately", "Remove and destroy infected plants", "Use copper-based sprays"],
        "prevention": ["Plant certified disease-free seed", "Remove cull piles", "Monitor weather for blight risk"],
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "description": " healthy potato plant with no signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Crop rotation", "Certified seed potatoes", "Good field drainage"],
    },
    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "description": " healthy raspberry cane with no signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Remove old fruiting canes", "Proper spacing for air circulation", "Monitor for aphids"],
    },
    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "description": " healthy soybean plant with no signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Rotate crops", "Monitor for aphids", "Maintain proper soil nutrition"],
    },
    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery mildew",
        "description": "A fungal disease causing white powdery growth on squash leaves and stems.",
        "symptoms": ["White powdery patches on leaves, stems, and fruit", "Leaves may curl or dry out", "Reduced photosynthesis and yield"],
        "treatment": ["Apply fungicide containing sulfur or potassium bicarbonate", "Remove affected leaves", "Improve air circulation"],
        "prevention": ["Plant in full sun with good spacing", "Avoid overhead watering", "Apply preventive fungicide sprays"],
    },
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf scorch",
        "description": "A fungal disease causing reddish-purple spots on strawberry leaves that enlarge with gray centers.",
        "symptoms": ["Reddish-purple circular spots on leaves", "Spots enlarge with grayish-white centers and dark margins", "Leaf death and premature senescence"],
        "treatment": ["Apply fungicide containing chlorothalonil or sulfur", "Remove affected leaves", "Improve field drainage"],
        "prevention": ["Rotate crops", "Remove and destroy infected plants", "Avoid overhead irrigation"],
    },
    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "description": " healthy strawberry plant with no signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Rotate crops", "Remove weeds", "Monitor for aphids and spider mites"],
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial spot",
        "description": "A bacterial disease causing small, dark, water-soaked spots on tomato leaves and fruit.",
        "symptoms": ["Small, dark brown, water-soaked spots on leaves", "Lesions with yellow halos", "Dark, raised spots on fruit", "Premature leaf drop"],
        "treatment": ["Apply copper-based bactericides", "Remove affected plant parts", "Rotate crops"],
        "prevention": ["Avoid overhead watering", "Remove plant debris", "Stake plants for better airflow"],
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early blight",
        "description": "A fungal disease causing dark spots with concentric rings on tomato leaves and fruit.",
        "symptoms": ["Small dark spots on lower leaves with concentric rings", "Lesions may expand and merge", "Yellowing of leaves above spots", "Fruit spots with dark concentric rings"],
        "treatment": ["Apply fungicide containing chlorothalonil or mancozeb", "Remove infected leaves", "Improve air circulation"],
        "prevention": ["Rotate crops", "Stake plants, mulch", "Water at soil level"],
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late blight",
        "description": "A highly destructive fungal disease causing rapid blighting of tomato plants.",
        "symptoms": ["Water-soaked lesions on leaves and fruit", "White fuzzy growth on underside in humid conditions", "Rapid plant death"],
        "treatment": ["Apply fungicide containing chlorothalonil or mancozeb immediately", "Remove and destroy infected plants", "Use copper-based sprays"],
        "prevention": ["Remove cull piles", "Rogue infected plants", "Monitor weather for blight risk"],
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "description": "A fungal disease causing yellow spots on upper leaf surfaces and fuzzy growth underneath.",
        "symptoms": ["Yellow spots on upper leaf surface", "Fuzzy gray-green growth on lower leaf surface", "Leaf yellowing and premature drop"],
        "treatment": ["Apply fungicide containing chlorothalonil or sulfur", "Improve greenhouse ventilation", "Remove affected leaves"],
        "prevention": ["Improve greenhouse ventilation", "Reduce humidity", "Rotate crops"],
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria leaf spot",
        "description": "A fungal disease causing numerous small spots with dark borders and white centers on tomato leaves.",
        "symptoms": ["Small circular spots with dark borders and white centers on leaves", "Spots may merge causing large blighted areas", "Yellowing of leaves above spots"],
        "treatment": ["Apply fungicide containing chlorothalonil or copper", "Remove affected leaves", "Rotate crops"],
        "prevention": ["Rotate crops", "Remove crop debris", "Stake plants for airflow"],
    },
    "Tomato___Spider_mites": {
        "plant": "Tomato",
        "disease": "Spider mites",
        "description": "A microscopic pest causing fine webbing and stippling on tomato leaves.",
        "symptoms": ["Fine webbing on leaves and between leaflets", "Tiny yellow/white stippling on leaf surfaces", "Leaf curling and premature drop"],
        "treatment": ["Apply miticide containing abamectin or bifenazate", "Introduce predatory mites", "Remove severely infested leaves"],
        "prevention": ["Maintain humidity control", "Introduce beneficial insects", "Monitor undersides of leaves"],
    },
    "Tomato___Tomato_yellow_leaf_curl_virus": {
        "plant": "Tomato",
        "disease": "Tomato yellow leaf curl virus",
        "description": "A viral disease causing leaf curling, yellowing, and stunted growth transmitted by whiteflies.",
        "symptoms": ["Leaf curling and upward rolling", "Yellowing and mottling of leaves", "Stunted plant growth and reduced yield"],
        "treatment": ["No cure, remove infected plants", "Control whiteflies with insecticides", "Use virus-resistant varieties"],
        "prevention": ["Control whitefly vectors", "Use reflective mulches", "Install row covers"],
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "description": " healthy tomato plant with no signs of disease.",
        "symptoms": [],
        "treatment": [],
        "prevention": ["Rotate crops", "Stake plants", "Monitor for pests and diseases"],
    },
}

def load_model():
    import torch
    # Download from: https://huggingface.co/BiernyVR/crop-disease-classifier
    checkpoint = torch.load("model_weights.pth", map_location="cpu")
    sd = checkpoint["model_state_dict"]
    classes = checkpoint["classes"]

    model = torchvision.models.efficientnet_v2_s(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(1280, len(classes)),
    )
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@app.on_event("startup")
def startup():
    app.state.model = load_model()


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


@app.get("/")
def get_upload_form():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Plant Disease Detection</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }
            .form-group { margin-bottom: 20px; }
            input[type=file] { padding: 10px; }
            button { padding: 10px 20px; font-size: 16px; }
            .result-section { margin-top: 20px; padding: 20px; background: #f5f5f5; display: none; }
            .plant-name { font-size: 24px; font-weight: bold; color: #333; margin: 10px 0; }
            .disease-card { background: white; padding: 15px; border-radius: 8px; margin-top: 10px; }
            .severity { color: #666; font-size: 14px; margin-top: 10px; }
            .toggle { cursor: pointer; color: #0066cc; margin-top: 5px; }
        </style>
    </head>
    <body>
        <h1>Plant Disease Detection</h1>
        <!-- Note: This model uses stock ImageNet EfficientNet-B0. Accurate plant disease classification requires a fine-tuned checkpoint. -->
        <div class="form-group" style="font-size: 12px; color: #666; margin-bottom: 20px;">
            ⚠ Model is stock ImageNet EfficientNet-B0. Accuracy requires a fine-tuned plant disease checkpoint.
        </div>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label>Select an image:</label>
                <input type="file" name="file" accept="image/*" required>
            </div>
            <div id="supportedCrops" class="form-group">
                <label>Supported crops:</label>
                <span class="toggle">Loading...</span>
            </div>
            <button type="submit">Detect Disease</button>
        </form>
        <div class="result-section" id="resultSection">
            <div class="plant-name" id="plantName"></div>
            <div class="disease-card" id="diseaseCard">
                <div id="diseaseInfo"></div>
                <div class="severity" id="severityInfo"></div>
                <button class="toggle" onclick="expandTreatment()">Show Treatment</button>
                <div id="treatmentSection" style="display:none; margin-top: 10px;"></div>
                <button class="toggle" onclick="expandPrevention()">Show Prevention</button>
                <div id="preventionSection" style="margin-top: 10px; display:none;"></div>
            </div>
            <button onclick="tryAnother()" style="margin-top: 10px; padding: 8px 16px;">Try another</button>
        </div>

        <script>
            async function loadSupportedCrops() {
                const response = await fetch('/diseases');
                const data = await response.json();
                const cropsContainer = document.getElementById('supportedCrops');
                cropsContainer.innerHTML = '';
                data.diseases.forEach(disease => {
                    const label = document.createElement('label');
                    label.className = 'toggle';
                    label.innerHTML = `${disease.plant}: ${disease.disease}`;
                    cropsContainer.appendChild(label);
                });
            }

            loadSupportedCrops();

            function showResult(data) {
                const section = document.getElementById('resultSection');
                const plantName = document.getElementById('plantName');
                const diseaseInfo = document.getElementById('diseaseInfo');
                const severityInfo = document.getElementById('severityInfo');
                const treatmentSection = document.getElementById('treatmentSection');
                const preventionSection = document.getElementById('preventionSection');

                plantName.innerHTML = `🌿 ${data.plant}`;
                diseaseInfo.innerHTML = `<strong>${data.disease}</strong><br>${data.description}`;

                severityInfo.innerHTML = `<strong>Confidence:</strong> ${data.confidence.toFixed(2)}`;

                treatmentSection.innerHTML = `<strong>Treatment:</strong><br>${data.treatment.join('<br>')}`;
                preventionSection.innerHTML = `<strong>Prevention:</strong><br>${data.prevention.join('<br>')}`;

                section.style.display = 'block';
            }

            function expandTreatment() {
                const btn = event.target;
                const section = document.getElementById('treatmentSection');
                if (section.style.display === 'none') {
                    section.style.display = 'block';
                    btn.innerHTML = 'Hide Treatment';
                } else {
                    section.style.display = 'none';
                    btn.innerHTML = 'Show Treatment';
                }
            }

            function expandPrevention() {
                const btn = event.target;
                const section = document.getElementById('preventionSection');
                if (section.style.display === 'none') {
                    section.style.display = 'block';
                    btn.innerHTML = 'Hide Prevention';
                } else {
                    section.style.display = 'none';
                    btn.innerHTML = 'Show Prevention';
                }
            }

            function tryAnother() {
                // Reload the page without full refresh
                location.reload();
            }

            // Handle form submission
            const form = document.querySelector('form');
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = form.querySelector('input[type=file]');
                const file = fileInput.files[0];
                if (!file) return;

                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                showResult(data);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if file.size is not None and file.size > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    with torch.no_grad():
        tensor = preprocess_image(image)
        outputs = app.state.model(tensor)
        probabilities = torch.softmax(outputs[0], dim=0)
        top_probabilities, top_indices = torch.topk(probabilities, k=5, dim=0)
        confidence = top_probabilities[0].item()
        predicted_idx_val = top_indices[0].item()
        predicted_class = CLASS_NAMES[predicted_idx_val]
    
    # Look up disease information — CLASS_NAMES now directly contains disease keys
    # e.g. "Tomato___Late_blight", "Apple___Apple_scab", etc.
    disease_key = predicted_class
    
    # Build top-k results
    top_k_results = []
    for prob_val, idx_val in zip(top_probabilities, top_indices):
        idx_item = idx_val.item()
        class_name = CLASS_NAMES[idx_item]
        if class_name in DISEASES_DB:
            d_info = DISEASES_DB[class_name]
            top_k_results.append({
                "rank": len(top_k_results) + 1,
                "plant": d_info["plant"],
                "disease": d_info["disease"],
                "probability": float(prob_val),
            })
        else:
            top_k_results.append({
                "rank": len(top_k_results) + 1,
                "plant": class_name.split(" - ")[0] if " - " in class_name else class_name,
                "disease": class_name,
                "probability": float(prob_val),
            })
    
    if disease_key in DISEASES_DB:
        disease_info = DISEASES_DB[disease_key]
        result = {
            "plant": disease_info["plant"],
            "disease": disease_info["disease"],
            "confidence": float(confidence),
            "description": disease_info["description"],
            "symptoms": disease_info["symptoms"],
            "treatment": disease_info["treatment"],
            "prevention": disease_info["prevention"],
            "top_k": top_k_results,
        }
    else:
        # Disease info not available — fallback
        result = {
            "plant": predicted_class.split(" - ")[0] if " - " in predicted_class else predicted_class,
            "disease": predicted_class,
            "confidence": float(confidence),
            "description": "Disease information not available for this prediction.",
            "symptoms": [],
            "treatment": [],
            "prevention": [],
            "top_k": top_k_results,
        }
    
    return result


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/diseases")
def list_diseases():
    """Endpoint listing all supported plant diseases with descriptions."""
    diseases_list = []
    for key, info in DISEASES_DB.items():
        diseases_list.append({
            "plant": info["plant"],
            "disease": info["disease"],
            "description": info["description"],
        })
    return {"diseases": diseases_list}