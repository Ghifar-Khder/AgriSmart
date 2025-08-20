import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load your pre-trained model
model = keras.models.load_model(r"models\PlantVillage-models\efficientnetB0_model99,47%.keras\efficientnetB0_model99,47%.keras")

# Hardcoded class names from your dataset
class_names = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy',
    'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

IMG_SIZE = (224, 224)

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Classifier")
        self.root.geometry("800x700")  # interface Width x Height        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
        
        # Create main container
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        self.header = ttk.Label(self.main_frame, text="Plant Disease Classifier", style='Header.TLabel')
        self.header.pack(pady=10)
        
        # Instructions
        self.instructions = ttk.Label(self.main_frame, 
                                    text="Click on the image area or 'Browse' button to select an image",
                                    wraplength=400)
        self.instructions.pack(pady=10)
        
        # Image display area
        self.image_frame = ttk.Frame(self.main_frame, relief=tk.SUNKEN, width=400, height=300)
        self.image_frame.pack(pady=20)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder image
        self.placeholder_img = Image.new('RGB', (400, 300), color='#e0e0e0')
        self.placeholder_photo = ImageTk.PhotoImage(self.placeholder_img)
        self.image_label.config(image=self.placeholder_photo)
        
        # Browse button
        self.browse_button = ttk.Button(self.main_frame, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=10)
        
        # Prediction area
        self.prediction_frame = ttk.Frame(self.main_frame)
        self.prediction_frame.pack(fill=tk.X, pady=20)
        
        self.prediction_label = ttk.Label(self.prediction_frame, text="Prediction will appear here", wraplength=600)
        self.prediction_label.pack()
        
        # Confidence bar
        self.confidence_frame = ttk.Frame(self.main_frame)
        self.confidence_frame.pack(fill=tk.X, pady=10)

        # Create a CENTER container inside confidence_frame
        self.center_container = ttk.Frame(self.confidence_frame)
        self.center_container.pack(expand=True)  # This centers everything inside

        # Now pack widgets into the center_container (instead of confidence_frame)
        self.confidence_label = ttk.Label(self.center_container, text="Confidence:")
        self.confidence_label.pack(side=tk.LEFT)

        self.confidence_bar = ttk.Progressbar(self.center_container, length=300)
        self.confidence_bar.pack(side=tk.LEFT, padx=10)

        self.confidence_value = ttk.Label(self.center_container, text="0%")
        self.confidence_value.pack(side=tk.LEFT)
        
        # Bind click event to image label
        self.image_label.bind("<Button-1>", self.ask_open_file)
    
    def ask_open_file(self, event=None):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.process_image(file_path)
    
    def browse_image(self):
        self.ask_open_file()
    
    def process_image(self, file_path):
        try:
            # Load and display image
            img = Image.open(file_path)
            img.thumbnail((400, 300))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep reference
            
            # Preprocess and predict
            img_array = self.preprocess_image(file_path)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Format the class name for better display
            formatted_name = self.format_class_name(class_names[predicted_class])
            
            # Update UI with prediction
            self.prediction_label.config(
                text=f"Prediction: {formatted_name}\nConfidence: {confidence*100:.2f}%"
            )
            self.confidence_bar['value'] = confidence * 100
            self.confidence_value.config(text=f"{confidence*100:.1f}%")
            
        except Exception as e:
            self.prediction_label.config(text=f"Error: {str(e)}")
    
    def format_class_name(self, class_name):
        """Format the class name for better readability"""
        # Replace '___' with ': ' and '_' with spaces
        formatted = class_name.replace('___', ': ').replace('_', ' ')
        # Handle special cases
        formatted = formatted.replace('(including sour)', '(incl. sour)')
        formatted = formatted.replace('maize', 'corn')
        formatted = formatted.replace('Haunglongbing', 'Huanglongbing')
        return formatted
    
    def preprocess_image(self, file_path):
        # Load and preprocess the image for the model
        img = tf.keras.utils.load_img(file_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        return img_array

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()