# **PBAT: Protein Binding Affinity using Transformers**

## Overview
PBAT (Protein Binding Affinity using Transformers) is a state-of-the-art model for predicting protein-ligand binding affinity, leveraging advanced transformer architectures. Designed to support drug discovery and agrochemical applications, PBAT excels in handling large datasets, extracting meaningful insights from complex protein-ligand interactions, and delivering precise binding affinity predictions.

## Features
- **Transformer-Based Architecture**: Utilizes attention mechanisms for capturing long-range dependencies in protein-ligand data.
- **High Precision**: Achieves Mean Squared Error (MSE) of **0.7047**, outperforming traditional methods.
- **Customizable Hyperparameters**: Flexible architecture for fine-tuning to specific datasets or use cases.
- **Integration-Ready**: Seamlessly integrates with other Bindwell technologies, such as AffinityLM and DrugDiscoveryGPT.

---

## **Installation**

### Requirements
PBAT requires the following dependencies:
- Python >= 3.8
- PyTorch >= 1.12.0
- Transformers >= 4.0.0
- RDKit
- NumPy
- Scikit-learn
- Pandas

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Bindwell/PBAT.git
   cd PBAT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install PBAT:
   ```bash
   python setup.py install
   ```

4. Verify installation:
   ```bash
   python -m PBAT.test
   ```

---

## **Model Architecture**

PBAT incorporates a transformer-based model with the following configuration:

### **Hyperparameters**
- **Embedding Dimension**: 768
- **Linear Layer Dimension**: 1472
- **Number of Attention Layers**: 3
- **Attention Heads**: 3
- **Dropout Rate**: 0.05
- **Learning Rate**: \(1 \times 10^{-6}\)
- **Batch Size**: 7
- **Epochs**: 250
- **Early Stopping Patience**: 10

### **Input Features**
PBAT processes protein-ligand pair data as input:
1. **Protein Sequences**: Encoded using one-hot encoding or learned embeddings.
2. **Ligand Representations**: SMILES strings converted into molecular embeddings.

---

## **Usage**

### **Training**
To train PBAT on your dataset:
1. Prepare your dataset in CSV format with columns:
   - `protein_sequence`
   - `ligand_SMILES`
   - `binding_affinity`
2. Run the training script:
   ```bash
   python train.py --data_path path/to/dataset.csv --save_dir path/to/save_model
   ```

### **Inference**
Use a trained model for prediction:
```python
from PBAT import PBATModel

# Load model
model = PBATModel.load_from_checkpoint("path/to/checkpoint.pt")

# Make prediction
protein = "MKTLLILAVGVLLAVMLASVQ"  # example protein sequence
ligand = "CCOCC"  # example SMILES
predicted_affinity = model.predict(protein, ligand)
print(f"Predicted Binding Affinity: {predicted_affinity}")
```

---

## **Validation**

PBAT has been validated on multiple benchmark datasets, achieving:
- **MSE**: 0.7047
- **Correlation Coefficient**: 0.92

For detailed validation metrics and model performance, refer to [**Model Performance Documentation**](docs/Validation.md).

---

## **Contributing**
We welcome contributions to improve PBAT. Please read our [**Contribution Guide**](docs/CONTRIBUTING.md) for guidelines on reporting issues, feature requests, and submitting pull requests.

---

## **License**
PBAT is distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

For further inquiries or support, contact us at [**Bindwell Support**](mailto:support@bindwell.ai).
