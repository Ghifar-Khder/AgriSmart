import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import joblib
import pandas as pd
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
class DiseaseClassifierTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Load the trained model
        try:
            model_path = r"models\PlantVillage-models\efficientnetB0_model99,47%.keras"
            self.model = keras.models.load_model(model_path)
            print("Disease model loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load disease model: {str(e)}", file=sys.stderr)
            tk.messagebox.showerror("Error", f"Failed to load disease model: {str(e)}")
        
        # class names from the dataset
        self.class_names = [
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
        self.IMG_SIZE = (224, 224)
        
#_____________________________________________________________
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
        
        # Create UI
        self.create_widgets()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Header
        self.header = ttk.Label(self, text="Plant Disease Classifier", style='Header.TLabel')
        self.header.pack(pady=10)
        
        # Instructions
        self.instructions = ttk.Label(self, 
                                    text="Click on the image area or 'Browse' button to select an image",
                                    wraplength=400)
        self.instructions.pack(pady=10)
        
        # Image display area
        self.image_frame = ttk.Frame(self, relief=tk.SUNKEN, width=400, height=300)
        self.image_frame.pack(pady=20)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder image
        self.placeholder_img = Image.new('RGB', (400, 300), color='#e0e0e0')
        self.placeholder_photo = ImageTk.PhotoImage(self.placeholder_img)
        self.image_label.config(image=self.placeholder_photo)
        
        # Browse button
        self.browse_button = ttk.Button(self, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=10)
        
        # Prediction area
        self.prediction_frame = ttk.Frame(self)
        self.prediction_frame.pack(fill=tk.X, pady=20)
        
        self.prediction_label = ttk.Label(self.prediction_frame, text="Prediction will appear here", wraplength=600)
        self.prediction_label.pack()
        
        # Confidence bar
        self.confidence_frame = ttk.Frame(self)
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
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Format the class name for better display
            formatted_name = self.format_class_name(self.class_names[predicted_class])
            
            # Update UI with prediction
            self.prediction_label.config(
                text=f"Prediction: {formatted_name}"
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
        img = tf.keras.utils.load_img(file_path, target_size=self.IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array


class YieldPredictorTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Redirect stderr to terminal
        sys.stderr = sys.__stderr__
        
        # Load ML model and preprocessor
        self.load_models()
        
        # Create UI
        self.create_widgets()
        
    def load_models(self):
        """Load the trained ML model and preprocessor"""
        try:
            model_path = r"models\yield_prediction-models\RandomForest_best.pkl"
            preprocessor_path = r"models\yield_prediction-models\preprocessor.pkl"
            
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            
            # Get the original feature names expected by the model
            self.original_feature_names = self.model.feature_names_in_
            
            print("Yield models loaded successfully")
            
        except Exception as e:
            print(f"ERROR: Failed to load yield model files: {str(e)}", file=sys.stderr)
            tk.messagebox.showerror("Error", f"Failed to load yield models: {str(e)}")
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding=10)
        input_frame.pack(fill=tk.X, pady=5)
        
        # Numerical inputs
        ttk.Label(input_frame, text="Average Rainfall (mm/year):").grid(row=0, column=0, sticky='w', pady=5)
        self.rainfall = ttk.Entry(input_frame)
        self.rainfall.grid(row=0, column=1, pady=5, sticky='ew')
        self.rainfall.insert(0, "500")
        
        ttk.Label(input_frame, text="Pesticides (tonnes):").grid(row=1, column=0, sticky='w', pady=5)
        self.pesticides = ttk.Entry(input_frame)
        self.pesticides.grid(row=1, column=1, pady=5, sticky='ew')
        self.pesticides.insert(0, "100")
        
        ttk.Label(input_frame, text="Average Temperature (°C):").grid(row=2, column=0, sticky='w', pady=5)
        self.temperature = ttk.Entry(input_frame)
        self.temperature.grid(row=2, column=1, pady=5, sticky='ew')
        self.temperature.insert(0, "20")
        
        # Dropdowns for categorical features
        ttk.Label(input_frame, text="Country:").grid(row=3, column=0, sticky='w', pady=5)
        self.country = ttk.Combobox(input_frame, values=[
            "Albania", "Algeria", "Angola", "Argentina", "Armenia", "Australia", "Austria", 
            "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Belarus", "Belgium", 
            "Botswana", "Brazil", "Bulgaria", "Burkina Faso", "Burundi", "Cameroon", 
            "Canada", "Central African Republic", "Chile", "Colombia", "Croatia", 
            "Denmark", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", 
            "Eritrea", "Estonia", "Finland", "France", "Germany", "Ghana", "Greece", 
            "Guatemala", "Guinea", "Guyana", "Haiti", "Honduras", "Hungary", "India", 
            "Indonesia", "Iraq", "Ireland", "Italy", "Jamaica", "Japan", "Kazakhstan", 
            "Kenya", "Latvia", "Lebanon", "Lesotho", "Libya", "Lithuania", "Madagascar", 
            "Malawi", "Malaysia", "Mali", "Mauritania", "Mauritius", "Mexico", 
            "Montenegro", "Morocco", "Mozambique", "Namibia", "Nepal", "Netherlands", 
            "New Zealand", "Nicaragua", "Niger", "Norway", "Pakistan", "Papua New Guinea", 
            "Peru", "Poland", "Portugal", "Qatar", "Romania", "Rwanda", "Saudi Arabia", 
            "Senegal", "Slovenia", "South Africa", "Spain", "Sri Lanka", "Sudan", 
            "Suriname", "Sweden", "Switzerland", "Tajikistan", "Thailand", "Tunisia", 
            "Turkey", "Uganda", "Ukraine", "United Kingdom", "Uruguay", "Zambia", "Zimbabwe"
        ], state="readonly")
        self.country.grid(row=3, column=1, pady=5, sticky='ew')
        self.country.current(0)
        
        ttk.Label(input_frame, text="Crop Type:").grid(row=4, column=0, sticky='w', pady=5)
        self.crop = ttk.Combobox(input_frame, values=[
            "Cassava", "Maize", "Plantains and others", "Potatoes", "Rice, paddy",
            "Sorghum", "Soybeans", "Sweet potatoes", "Wheat", "Yams"
        ], state="readonly")
        self.crop.grid(row=4, column=1, pady=5, sticky='ew')
        self.crop.current(0)
        
        # Predict button
        ttk.Button(input_frame, text="Predict Yield", command=self.predict).grid(
            row=5, column=0, columnspan=2, pady=10, sticky='ew')
        
        # Output section (simplified without interpretation)
        output_frame = ttk.LabelFrame(main_frame, text="Prediction Result", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.result = ttk.Label(output_frame, text="Enter values and click 'Predict'", 
                              font=('Arial', 12))
        self.result.pack(pady=10, expand=True)
        
        # Configure grid weights
        input_frame.columnconfigure(1, weight=1)
    
    def validate_inputs(self):
        """Validate all input fields and return error message if invalid"""
        try:
            float(self.rainfall.get())
        except ValueError:
            return "Rainfall must be a valid number"
        
        try:
            float(self.pesticides.get())
        except ValueError:
            return "Pesticides must be a valid number"
        
        try:
            float(self.temperature.get())
        except ValueError:
            return "Temperature must be a valid number"
            
        if not self.country.get():
            return "Please select a country"
            
        if not self.crop.get():
            return "Please select a crop type"
            
        return None
    
    def predict(self):
        """Make prediction using the ML model"""
        # Validate inputs first
        error_msg = self.validate_inputs()
        if error_msg:
            print(f"INPUT ERROR: {error_msg}", file=sys.stderr)
            self.result.config(text=f"Error: {error_msg}")
            return
            
        try:
            # Get input values
            input_data = {
                "average_rain_fall_mm_per_year": float(self.rainfall.get()),
                "pesticides_tonnes": float(self.pesticides.get()),
                "avg_temp": float(self.temperature.get()),
                "Area": self.country.get(),
                "Item": self.crop.get()
            }
            
            # Convert to DataFrame with proper column names
            input_df = pd.DataFrame([input_data], columns=[
                "average_rain_fall_mm_per_year",
                "pesticides_tonnes",
                "avg_temp",
                "Area",
                "Item"
            ])
            
            # Preprocess the input
            processed = self.preprocessor.transform(input_df)
            
            # Convert to DataFrame with the exact feature names the model expects
            processed_df = pd.DataFrame(processed, columns=self.original_feature_names)
            
            # Make prediction
            prediction = self.model.predict(processed_df)[0]
            
            # Update UI
            self.result.config(text=f"Predicted Yield: {prediction:,.2f} hg/ha")
            print(f"SUCCESS: Predicted yield: {prediction:,.2f} hg/ha")
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            print(f"PREDICTION ERROR: {error_msg}", file=sys.stderr)
            self.result.config(text="Error in prediction")
            import traceback
            traceback.print_exc(file=sys.stderr)


class CropProductivitySuite:
    def __init__(self, root):
        self.root = root
        self.root.title("Crop Productivity Suite")
        self.root.geometry("800x700")
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs - Disease Classifier first
        self.disease_tab = DiseaseClassifierTab(self.notebook)
        self.yield_tab = YieldPredictorTab(self.notebook)
        
        # Add tabs to notebook with Disease as first tab
        self.notebook.add(self.disease_tab, text="Plant Disease Classifier")
        self.notebook.add(self.yield_tab, text="Crop Yield Predictor")
        
        # Add a status bar
        self.status = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)


if __name__ == "__main__":
    root = tk.Tk()
    app = CropProductivitySuite(root)
    root.mainloop()