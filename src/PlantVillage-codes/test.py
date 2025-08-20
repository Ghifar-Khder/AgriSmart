import tensorflow as tf
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math

# Configuration
MODEL_PATH = r"models\PlantVillage-models\efficientnetB0_model99,47%.keras"
DATASET_DIRS = {
    "train": "data\PlantVillage-data\train",
    "val": "data\PlantVillage-data\val",
    "test": "data\PlantVillage-data\test"
}
OUTPUT_DIR = "evaluation_results"

def create_dataset(directory, batch_size=32):
    """Create a TensorFlow dataset from image directory"""
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,
        label_mode='categorical'
    )

def save_confusion_matrix_with_numbers(cm, class_names, model_name, dataset_name):
    """Save confusion matrix with clear numbers and class names"""
    n_classes = len(class_names)
    
    # Dynamic sizing - at least 20 inches, or 0.5 inch per class
    base_size = max(20, n_classes * 0.5)
    
    # Create figure with dynamic size
    plt.figure(figsize=(base_size, base_size))
    
    # Calculate appropriate font size for annotations
    annotation_size = max(6, min(12, 100/n_classes))  # Scale between 6-12 based on class count
    
    # Create heatmap with numbers
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        annot_kws={"size": annotation_size},
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"shrink": 0.75}
    )
    
    # Adjust title and labels
    plt.title(
        f'Confusion Matrix - {model_name} - {dataset_name}',
        fontsize=14,
        pad=20
    )
    plt.xlabel('Predicted', fontsize=12, labelpad=10)
    plt.ylabel('True', fontsize=12, labelpad=10)
    
    # Rotate labels and adjust appearance
    plt.xticks(
        rotation=90,
        ha='center',
        fontsize=8,
        fontfamily='monospace'
    )
    plt.yticks(
        rotation=0,
        fontsize=8,
        fontfamily='monospace'
    )
    
    # Tight layout with extra padding
    plt.tight_layout(pad=3.0)
    
    # Save with high DPI
    cm_filename = f"{model_name}_{dataset_name}_confusion_matrix_with_numbers.png"
    cm_path = os.path.join(OUTPUT_DIR, cm_filename)
    
    plt.savefig(
        cm_path,
        bbox_inches='tight',
        dpi=300,
        transparent=False
    )
    plt.close()
    
    print(f"\nSaved confusion matrix with numbers to:\n{cm_path}")
    print(f"Image size: {base_size}×{base_size} inches at 300 DPI")
    
    # Additionally save a version with every nth label for overview
    if n_classes > 30:
        save_reduced_labels_matrix(cm, class_names, model_name, dataset_name)

def save_reduced_labels_matrix(cm, class_names, model_name, dataset_name):
    """Save a version with fewer labels but keeping all numbers"""
    n_classes = len(class_names)
    step = max(1, math.ceil(n_classes / 30))  # Show ~30 labels max
    
    plt.figure(figsize=(20, 20))
    
    # Create heatmap with all numbers but reduced labels
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        annot_kws={"size": 8},
        xticklabels=[class_names[i] if i % step == 0 else '' for i in range(n_classes)],
        yticklabels=[class_names[i] if i % step == 0 else '' for i in range(n_classes)],
    )
    
    plt.title(f'Confusion Matrix (Reduced Labels) - {model_name} - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    reduced_filename = f"{model_name}_{dataset_name}_confusion_matrix_reduced_labels.png"
    reduced_path = os.path.join(OUTPUT_DIR, reduced_filename)
    plt.savefig(reduced_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved reduced-labels confusion matrix to {reduced_path}")

def evaluate_model(model, dataset, dataset_name, model_name):
    """Evaluate model and save results"""
    print(f"\nEvaluating {dataset_name} dataset...")
    
    # Prepare outputs
    y_true = []
    y_pred = []
    class_names = dataset.class_names
    
    # Predict in batches with progress bar
    start_time = time.time()
    with tqdm(total=len(dataset), desc="Evaluation Progress", unit="batch") as pbar:
        for images, labels in dataset:
            y_true.extend(tf.argmax(labels, axis=1).numpy())
            y_pred.extend(tf.argmax(model.predict(images, verbose=0), axis=1))
            pbar.update(1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    eval_time = time.time() - start_time
    
    # Create output directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save confusion matrix with numbers
    save_confusion_matrix_with_numbers(cm, class_names, model_name, dataset_name)
    
    # Save metrics to text file
    metrics_content = f"""Evaluation Results - {model_name} - {dataset_name}
========================================
Evaluation Time: {eval_time:.2f} seconds
Total Samples: {len(y_true)}

Overall Metrics:
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}

Confusion matrices saved to {OUTPUT_DIR}
"""
    metrics_filename = f"{model_name}_{dataset_name}_metrics.txt"
    metrics_path = os.path.join(OUTPUT_DIR, metrics_filename)
    with open(metrics_path, 'w') as f:
        f.write(metrics_content)
    print(f"Saved metrics to {metrics_path}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'eval_time': eval_time
    }

def main():
    # Load model
    print(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    
    # Show dataset options
    print("\nAvailable datasets:")
    for i, (name, path) in enumerate(DATASET_DIRS.items()):
        print(f"{i+1}. {name} ({path})")
    
    # Get user selection
    while True:
        try:
            choice = int(input("\nEnter dataset number to evaluate (1-3): ")) - 1
            if 0 <= choice < len(DATASET_DIRS):
                break
            print("Please enter a number between 1 and 3")
        except ValueError:
            print("Please enter a valid number")
    
    dataset_name = list(DATASET_DIRS.keys())[choice]
    
    # Evaluate selected dataset
    print(f"\nPreparing {dataset_name} dataset...")
    dataset = create_dataset(DATASET_DIRS[dataset_name])
    
    results = evaluate_model(model, dataset, dataset_name, model_name)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Dataset: {dataset_name}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Evaluation Time: {results['eval_time']:.2f} seconds")

if __name__ == "__main__":
    main()