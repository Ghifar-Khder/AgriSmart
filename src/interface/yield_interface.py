import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd
import sys
from sklearn.compose import ColumnTransformer
import numpy as np
class YieldPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crop Yield Predictor")
        self.root.geometry("550x400")  # Reduced height since we removed interpretation
        
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
            
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"ERROR: Failed to load model files: {str(e)}", file=sys.stderr)
            self.root.destroy()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
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

if __name__ == "__main__":
    root = tk.Tk()
    app = YieldPredictorApp(root)
    root.mainloop()