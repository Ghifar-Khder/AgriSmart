import streamlit as st
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2
import io
import os
# ========================
# CONFIG
# ========================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../src/interface
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models")   # repo_root/models

DISEASE_MODEL_PATH = os.path.join(MODELS_DIR, "efficientnetB0_model_augmented.keras")
YIELD_MODEL_PATH = os.path.join(MODELS_DIR, "DecisionTree_best.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

CLASS_NAMES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy',
    'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

IMG_SIZE = (224, 224)

# ========================
# DISEASE INFO
# ========================
DISEASE_INFO = {
    'Apple___Apple_scab': {
        "Cause": "Fungus Venturia inaequalis.",
        "Symptoms": "Olive-green oily spots on leaves turning brown/black velvety; fruit becomes deformed with corky scabs.",
        "Prevention": "- Remove fallen leaves in autumn\n- Use resistant cultivars (Liberty, GoldRush)",
        "Treatment": "- Captan or Mancozeb (protective)\n- Myclobutanil (curative)",
    },
    'Apple___Black_rot': {
        "Cause": "Fungus Botryosphaeria obtusa.",
        "Symptoms": "Sunken brown fruit lesions forming concentric black rings; frog-eye leaf spots.",
        "Prevention": "- Prune and remove cankers immediately",
        "Treatment": "- Thiophanate-methyl during bud break",
    },
    'Apple___Cedar_apple_rust': {
        "Cause": "Fungus Gymnosporangium juniperi-virginianae.",
        "Symptoms": "Bright orange leaf spots; tube-like aecia structures underneath leaves.",
        "Prevention": "- Remove nearby juniper trees within 1–2 miles",
        "Treatment": "- Sterol Inhibitors (SI) like Myclobutanil",
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        "Cause": "Fungus Podosphaera clandestina.",
        "Symptoms": "White powdery growth on young leaves and shoots; leaf curling and dwarfing.",
        "Prevention": "- Improve air circulation via proper pruning",
        "Treatment": "- Wettable Sulfur\n- Horticultural oils",
    },
    'Grape___Black_rot': {
        "Cause": "Fungus Guignardia bidwellii.",
        "Symptoms": "Circular brown leaf spots; berries become black shriveled mummies.",
        "Prevention": "- Remove mummified fruits from vines and soil",
        "Treatment": "- Mancozeb from bud break to flowering",
    },
    'Grape___Esca_(Black_Measles)': {
        "Cause": "Fungal complex (Phaeomoniella chlamydospora, Phaeoacremonium aleophilum).",
        "Symptoms": "Tiger-stripe yellow/red leaf patterns; black measles-like fruit spotting.",
        "Prevention": "- Disinfect pruning tools\n- Protect pruning wounds with fungicidal sealant",
        "Treatment": "No fully effective chemical cure; management via vineyard replacement and sanitation.",
    },
    'Pepper,_bell___Bacterial_spot': {
        "Cause": "Bacterium Xanthomonas euvesicatoria.",
        "Symptoms": "Small water-soaked leaf spots turning brown with yellow halos; defoliation.",
        "Prevention": "- Certified disease-free seeds\n- 2-year crop rotation",
        "Treatment": "- Copper-based bactericides + Mancozeb",
    },
    'Potato___Early_blight': {
        "Cause": "Fungus Alternaria solani.",
        "Symptoms": "Dark brown lower-leaf lesions with concentric target-like rings.",
        "Prevention": "- Maintain good plant nutrition (especially nitrogen)",
        "Treatment": "- Chlorothalonil\n- Strobilurins",
    },
    'Potato___Late_blight': {
        "Cause": "Oomycete Phytophthora infestans.",
        "Symptoms": "Large water-soaked lesions turning dark brown/black; white cottony growth underneath in humid weather.",
        "Prevention": "- Remove cull piles\n- Use certified seed potatoes",
        "Treatment": "- Mancozeb (protective)\n- Metalaxyl (systemic)",
    },
    'Squash___Powdery_mildew': {
        "Cause": "Podosphaera xanthii or Erysiphe cichoracearum.",
        "Symptoms": "White powdery coating on leaves and stems; early leaf drying.",
        "Prevention": "- Plant resistant varieties\n- Ensure good sunlight and airflow",
        "Treatment": "- Wettable Sulfur\n- Potassium Bicarbonate\n- Quintec",
    },
    'Strawberry___Leaf_scorch': {
        "Cause": "Fungus Diplocarpon earlianum.",
        "Symptoms": "Irregular purple spots merging into scorched brown leaves.",
        "Prevention": "- Remove old infected leaves\n- Avoid overhead irrigation",
        "Treatment": "- Captan\n- Thiophanate-methyl",
    },
    'Tomato___Bacterial_spot': {
        "Cause": "Xanthomonas spp.",
        "Symptoms": "Small black spots with yellow halos; raised scabby fruit lesions.",
        "Prevention": "- Heat-treated seeds\n- Avoid working in wet fields",
        "Treatment": "- Copper hydroxide + Mancozeb",
    },
    'Tomato___Early_blight': {
        "Cause": "Alternaria linariae.",
        "Symptoms": "Concentric target-like lesions on lower leaves first.",
        "Prevention": "- Crop rotation\n- Remove infected debris",
        "Treatment": "- Chlorothalonil\n- Azoxystrobin",
    },
    'Tomato___Late_blight': {
        "Cause": "Phytophthora infestans.",
        "Symptoms": "Dark green water-soaked lesions rapidly turning black; white mold underneath in mornings.",
        "Prevention": "- Proper spacing\n- Remove infected plants immediately",
        "Treatment": "- Metalaxyl\n- Propamocarb",
    },
    'Tomato___Leaf_Mold': {
        "Cause": "Passalora fulva.",
        "Symptoms": "Yellow upper-leaf spots; olive-brown velvety growth underneath.",
        "Prevention": "- Reduce greenhouse humidity\n- Improve ventilation",
        "Treatment": "- Increase airflow\n- Use preventive fungicides if necessary",
    },
    'Tomato___Septoria_leaf_spot': {
        "Cause": "Septoria lycopersici.",
        "Symptoms": "Small circular gray-centered spots with tiny black fruiting bodies.",
        "Prevention": "- Remove plant debris\n- Avoid overhead irrigation",
        "Treatment": "- Copper fungicides\n- Chlorothalonil",
    },
    'Tomato___Target_Spot': {
        "Cause": "Corynespora cassiicola.",
        "Symptoms": "Numerous small target-like lesions; sunken fruit spots.",
        "Prevention": "- Crop rotation\n- Improve airflow",
        "Treatment": "- Chlorothalonil\n- Appropriate fungicides",
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        "Cause": "Arthropod Tetranychus urticae.",
        "Symptoms": "Fine yellow stippling; webbing; bronzed leaves under heavy infestation.",
        "Prevention": "- Monitor with yellow sticky cards\n- Maintain humidity balance",
        "Treatment": "- Phytoseiulus persimilis (biocontrol)\n- Abamectin (acaricide)",
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        "Cause": "Virus transmitted by whitefly (Bemisia tabaci).",
        "Symptoms": "Upward leaf curling; yellowing margins; severe stunting.",
        "Prevention": "- Insect-proof netting (<500 micron)\n- Whitefly control\n- Resistant hybrids",
        "Treatment": "No chemical cure; remove infected plants immediately.",
    },
    'Tomato___Tomato_mosaic_virus': {
        "Cause": "Mechanical transmission virus (ToMV).",
        "Symptoms": "Light/dark green mosaic pattern; leaf deformation.",
        "Prevention": "- Disinfect tools\n- Avoid smoking near plants\n- Certified seeds",
        "Treatment": "No chemical cure; strict sanitation required.",
    },
}


# ========================
# LOAD MODELS
# ========================
@st.cache_resource
def load_disease_model():
    return keras.models.load_model(DISEASE_MODEL_PATH)

@st.cache_resource
def load_yield_models():
    model = joblib.load(YIELD_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = model.feature_names_in_
    return model, preprocessor, feature_names

disease_model = load_disease_model()
yield_model, preprocessor, feature_names = load_yield_models()


# ========================
# HELPER FUNCTIONS
# ========================
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def format_class_name(name):
    formatted = name.replace('___', ': ').replace('_', ' ')
    formatted = formatted.replace('(including sour)', '')
    formatted = formatted.replace('maize', 'corn')
    formatted = formatted.replace('Haunglongbing', 'Huanglongbing')
    return formatted


# ========================
# GRAD-CAM
# ========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    efficientnet_submodel = model.get_layer("efficientnetb0")

    conv_output_model = keras.Model(
        inputs=efficientnet_submodel.input,
        outputs=efficientnet_submodel.get_layer(last_conv_layer_name).output
    )

    dense_1_layer = model.get_layer("dense_1")
    dense_weights, dense_bias = dense_1_layer.get_weights()

    conv_input = keras.Input(shape=conv_output_model.output.shape[1:])
    x = conv_input
    for layer in model.layers:
        if layer.name in ["global_average_pooling2d", "dense", "dropout"]:
            x = layer(x)

    logits_output = keras.layers.Lambda(
        lambda inp: tf.matmul(inp, dense_weights) + dense_bias
    )(x)

    logit_model = keras.Model(inputs=conv_input, outputs=logits_output)

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs = conv_output_model(inputs, training=False)
        tape.watch(conv_outputs)
        logits = logit_model(conv_outputs, training=False)
        pred_index = tf.argmax(logits[0])
        class_channel = logits[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap.numpy()
    p5  = np.percentile(heatmap, 5)
    p95 = np.percentile(heatmap, 95)
    if p95 > p5:
        heatmap = np.clip((heatmap - p5) / (p95 - p5), 0, 1)
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap = np.power(heatmap, 0.7)
    return heatmap

def overlay_gradcam(original_img, heatmap):
    img_array = np.array(original_img.resize(IMG_SIZE))
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img_array, 0.7, heatmap_colored, 0.3, 0)
    return heatmap_colored, superimposed


# ========================
# CACHED INFERENCE
# ========================
# Runs the classifier + Grad-CAM ONCE per uploaded image and caches the
# result by the raw file bytes. Without this, Streamlit re-runs this whole
# script (including the model + Grad-CAM) on every single widget interaction
# on the page -- e.g. clicking the Cause/Symptoms/Prevention/Treatment
# selector -- which is why the heatmap appeared to "refresh" and the page
# would flash blank while it recomputed.
@st.cache_data(show_spinner=False)
def run_inference(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = preprocess_image(img)

    predictions = disease_model.predict(img_array)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    heatmap = make_gradcam_heatmap(img_array, disease_model)
    _, superimposed = overlay_gradcam(img, heatmap)

    return predicted_class, confidence, superimposed


# ========================
# PAGE CONFIG + STYLING
# ========================
st.set_page_config(
    page_title="Crop Productivity",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: linear-gradient(180deg, #f4faf6 0%, #eef6f0 100%);
}

.main-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    background: linear-gradient(90deg, #1b4332, #2d6a4f, #52b788);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.1rem;
}

.subtitle {
    text-align: center;
    color: #52796f;
    font-size: 1.05rem;
    margin-bottom: 1.8rem;
}

.section-label {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    color: #1b4332;
    margin-bottom: 0.4rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #ffffff;
    padding: 6px;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(27,67,50,0.06);
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    border-radius: 12px;
    padding: 0 22px;
    font-weight: 600;
    font-family: 'Poppins', sans-serif;
    color: #52796f;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #1b4332, #40916c) !important;
    color: white !important;
}

.result-card-healthy, .result-card-disease {
    border-radius: 18px;
    padding: 1.5rem 1.7rem;
    margin-bottom: 0.8rem;
}
.result-card-healthy {
    background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
    border-left: 6px solid #2d6a4f;
}
.result-card-disease {
    background: linear-gradient(135deg, #ffedd8, #ffdca8);
    border-left: 6px solid #e85d04;
}

.result-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
    color: #1b1b1b;
    margin-bottom: 0.5rem;
}

.confidence-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    background: #1b4332;
    color: white;
    font-weight: 600;
    font-size: 0.85rem;
}

.metric-box {
    background: linear-gradient(135deg, #1b4332, #2d6a4f);
    color: white;
    border-radius: 20px;
    padding: 2.2rem;
    text-align: center;
    box-shadow: 0 10px 28px rgba(27,67,50,0.25);
}
.metric-box .label {
    letter-spacing: 1px;
    font-size: 0.85rem;
    opacity: 0.8;
    font-weight: 600;
}
.metric-box .value {
    font-family: 'Poppins', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0.3rem 0;
}
.metric-box .caption {
    opacity: 0.85;
    font-size: 0.95rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1b4332, #2d6a4f);
}
section[data-testid="stSidebar"] * { color: #f1faee !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }

div.stButton > button {
    background: linear-gradient(90deg, #1b4332, #40916c);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-family: 'Poppins', sans-serif;
    transition: all 0.2s ease;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(27,67,50,0.3);
    color: white;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #40916c, #95d5b2);
}

[data-testid="stFileUploader"] {
    border-radius: 16px;
}

div[role="radiogroup"] {
    gap: 6px;
}
div[role="radiogroup"] label {
    background: #ffffff;
    border: 1px solid #d8e5dc;
    border-radius: 20px;
    padding: 6px 14px;
    margin-right: 4px;
    font-weight: 500;
}
div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, #1b4332, #40916c);
    border-color: #1b4332;
}
div[role="radiogroup"] label:has(input:checked) p {
    color: white !important;
}

.info-panel {
    background: #ffffff;
    border: 1px solid #e3ede6;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-top: 0.6rem;
    color: #333;
    line-height: 1.6;
}

/* Force readable widget colors regardless of the browser/OS light-dark theme */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-baseweb="input"] input,
[data-baseweb="select"] input {
    background-color: #ffffff !important;
    color: #1b1b1b !important;
}
[data-testid="stNumberInput"] > div,
[data-baseweb="input"],
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: #d8e5dc !important;
}
[data-baseweb="select"] div,
[data-baseweb="select"] span,
[data-baseweb="select"] p {
    color: #1b1b1b !important;
}
[data-baseweb="popover"] li,
[role="listbox"] {
    background-color: #ffffff !important;
    color: #1b1b1b !important;
}
[data-testid="stWidgetLabel"] p {
    color: #1b4332 !important;
    font-weight: 500;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #6b8577 !important;
}
[data-testid="stNumberInput"] button svg {
    fill: #1b4332 !important;
}
div[data-testid="stTooltipIcon"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)


# ========================
# SIDEBAR
# ========================
with st.sidebar:
    st.markdown("## 🌿 Crop Productivity")
    st.caption("AI-powered plant health & yield insights")
    st.markdown("---")
    st.markdown("#### 🌱 Disease Classifier")
    st.markdown("Upload a leaf photo to identify one of 38 plant health conditions, with a Grad-CAM view of what the model focused on.")
    st.markdown("#### 🌾 Yield Predictor")
    st.markdown("Enter climate and crop details to estimate expected yield in hg/ha.")
    st.markdown("---")
    st.caption("Ghifar Khder")


# ========================
# HEADER
# ========================
st.markdown('<div class="main-title">🌿 Crop Productivity</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect plant diseases and predict crop yield</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🌱  Plant Disease Classifier", "🌾  Crop Yield Predictor"])

# ------------------------
# TAB 1 — DISEASE CLASSIFIER
# ------------------------
with tab1:
    st.markdown('<div class="section-label">Upload a leaf image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with st.spinner("Analyzing leaf..."):
            predicted_class, confidence, superimposed = run_inference(image_bytes)

        disease_name = CLASS_NAMES[predicted_class]
        formatted_name = format_class_name(disease_name)
        is_healthy = "healthy" in disease_name.lower()

        st.markdown("<br>", unsafe_allow_html=True)
        col_img, col_result = st.columns([1, 1.3], gap="large")

        with col_img:
            st.image(img, use_container_width=True, caption="Uploaded Leaf")

        with col_result:
            card_class = "result-card-healthy" if is_healthy else "result-card-disease"
            icon = "✅" if is_healthy else "⚠️"
            st.markdown(f"""
            <div class="{card_class}">
                <div class="result-title">{icon} {formatted_name}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">📖 Disease Details</div>', unsafe_allow_html=True)
            info = DISEASE_INFO.get(disease_name, {})
            section = st.radio(
                "Select information type",
                ["🦠 Cause", "🔍 Symptoms", "🛡️ Prevention", "💊 Treatment"],
                horizontal=True,
                label_visibility="collapsed",
            )
            key = section.split(" ", 1)[1]
            if key in info:
                content_html = info[key].replace("\n", "<br>")
            elif is_healthy:
                content_html = "🌿 No issues detected — this leaf appears healthy."
            else:
                content_html = f"No {key.lower()} information available yet for this condition."
            st.markdown(f'<div class="info-panel">{content_html}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-label">🔍 Grad-CAM — where the model looked</div>', unsafe_allow_html=True)
        st.caption("Highlights the leaf regions the model focused on when making its prediction.")

        gc1, gc2 = st.columns(2)
        with gc1:
            st.image(img.resize(IMG_SIZE), caption="Original Image", use_container_width=True)
        with gc2:
            st.image(superimposed, caption="Model Focus Area", use_container_width=True)

# ------------------------
# TAB 2 — YIELD PREDICTOR
# ------------------------
with tab2:
    st.markdown('<div class="section-label">Enter conditions to estimate yield</div>', unsafe_allow_html=True)

    with st.container(border=True):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**🌦️ Climate Conditions**")
            rainfall = st.number_input(
                "🌧️ Average Rainfall (mm/year)",
                min_value=0.0, max_value=4000.0, value=500.0, step=10.0,
                help="Average annual rainfall for the region, in millimeters. Typical range: 300–2000 mm/year.",
            )
            st.caption("Average yearly rainfall for the region, in mm. Typical range: 300–2000.")

            temperature = st.number_input(
                "🌡️ Average Temperature (°C)",
                min_value=-10.0, max_value=50.0, value=20.0, step=0.5,
                help="Average yearly temperature for the region, in degrees Celsius.",
            )
            st.caption("Average yearly temperature for the region, in °C.")

            pesticides = st.number_input(
                "🧪 Pesticides Used (tonnes)",
                min_value=0.0, max_value=10000.0, value=100.0, step=10.0,
                help="Total pesticide use for the region/crop, in tonnes per year.",
            )
            st.caption("Total pesticide use for the region/crop, in tonnes per year.")

        with col2:
            st.markdown("**🌍 Location & Crop**")
            country = st.selectbox(
                "📍 Country",
                ["Albania","Algeria","Angola","Argentina","Armenia","Australia","Austria",
                "Azerbaijan","Bahamas","Bahrain","Bangladesh","Belarus","Belgium","Botswana","Brazil","Bulgaria",
                "Burkina Faso","Burundi","Cameroon","Canada","Central African Republic","Chile","Colombia","Croatia",
                "Denmark","Dominican Republic","Ecuador","Egypt","El Salvador","Eritrea","Estonia","Finland","France",
                "Germany","Ghana","Greece","Guatemala","Guinea","Guyana","Haiti","Honduras","Hungary","India",
                "Indonesia","Iraq","Ireland","Italy","Jamaica","Japan","Kazakhstan","Kenya","Latvia","Lebanon",
                "Lesotho","Libya","Lithuania","Madagascar","Malawi","Malaysia","Mali","Mauritania","Mauritius",
                "Mexico","Montenegro","Morocco","Mozambique","Namibia","Nepal","Netherlands","New Zealand",
                "Nicaragua","Niger","Norway","Pakistan","Papua New Guinea","Peru","Poland","Portugal","Qatar",
                "Romania","Rwanda","Saudi Arabia","Senegal","Slovenia","South Africa","Spain","Sri Lanka","Sudan",
                "Suriname","Sweden","Switzerland","Tajikistan","Thailand","Tunisia","Turkey","Uganda","Ukraine",
                "United Kingdom","Uruguay","Zambia","Zimbabwe"],
                help="Country the yield estimate is for.",
            )
            st.caption("Country the yield estimate is for.")
            crop = st.selectbox(
                "🌾 Crop Type",
                ["Cassava","Maize","Plantains and others","Potatoes","Rice, paddy",
                "Sorghum","Soybeans","Sweet potatoes","Wheat","Yams"],
                help="Crop the yield estimate is for.",
            )
            st.caption("Crop the yield estimate is for.")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🔍 Predict Yield", use_container_width=True)

    if predict_clicked:
        input_data = pd.DataFrame([{
            "average_rain_fall_mm_per_year": rainfall,
            "pesticides_tonnes": pesticides,
            "avg_temp": temperature,
            "Area": country,
            "Item": crop
        }])
        processed = preprocessor.transform(input_data)
        processed_df = pd.DataFrame(processed, columns=feature_names)
        prediction = yield_model.predict(processed_df)[0]
        rounded_prediction = int(round(prediction, -1))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">PREDICTED YIELD</div>
            <div class="value">{rounded_prediction:,} hg/ha</div>
            <div class="caption">for {crop} in {country}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown('<p style="text-align:center;color:#95a99c;font-size:0.85rem;">Crop Productivity · GHIFAR_KHDER</p>', unsafe_allow_html=True)

# streamlit run app2.py
