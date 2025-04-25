# Diabetic_Retinopathy_Severity_Classification
🧠 DeepRetinaX  
An EfficientNet-Based Deep Learning Model for Diabetic Retinopathy Severity Classification

---

🩺 Overview

DeepRetinaX is a deep learning solution designed to automatically classify the severity of Diabetic Retinopathy (DR) using fundus images.
Built with EfficientNet-B0, the model applies a combination of:

- ✅ Transfer Learning (using timm)
- ✅ Class-balanced Loss
- ✅ Data Augmentation
- ✅ Comprehensive Evaluation Metrics (Accuracy, MAE, RMSE, Kappa)

This project is structured for easy execution in Google Colab and supports real-time visualization and model evaluation.

---

📂 Folder Structure

DeepRetinaX/
├── efficientnet_training.py     # Full training pipeline  
├── model.py                     # Modular EfficientNet-B0 model  
├── requirements.txt             # All dependencies  
├── README.md                    # Project overview  
└── efficientnet_b0_best.pth     # (Optional) Saved model checkpoint  

---

🗃️ Dataset

- Source: IDRiD Dataset (https://idrid.grand-challenge.org/)  
- Focus: Disease Grading (5 Classes: 0 → 4)  
- Organize your dataset as:

  /Output/train/0/  
  /Output/train/1/  
  /Output/train/2/  
  /Output/train/3/  
  /Output/train/4/  

  and  

  /Output/val/[0-4]/  

---

🚀 Training Instructions

1. Mount your Google Drive in Colab  
2. Clone this repo or upload the files  
3. Make sure your data is in the correct format  
4. Run the training script:

   python efficientnet_training.py

During training, the script will:
- Track accuracy, MAE, RMSE, and Kappa score  
- Apply early stopping  
- Save the best model automatically to your Google Drive  

---

📊 Evaluation Metrics

Metric         | Meaning
-------------- | ----------------------------------------------
Accuracy       | Percentage of correct predictions
MAE            | Avg. absolute error between class predictions
RMSE           | Root mean squared error
Cohen’s Kappa  | Inter-rater reliability (balanced accuracy)

---

💡 Key Features

- 🔎 Transfer Learning using EfficientNet  
- 🧪 Real-time metrics and validation  
- ⚖️ Class imbalance handling via CrossEntropyLoss(weight=...)  
- 🧠 Modular architecture (clean imports)  

---

🛠 Requirements

Install all dependencies with:

pip install -r requirements.txt

> All packages are supported on Google Colab.

---

👨‍🎓 Author

Afridi Siddiqui  
Roll No: 21BT8033  
Department of Biotechnology  
NIT Durgapur  
Final Year B.Tech Project (2024–2025)

---

📜 License

This project is open-source under the MIT License.  
Feel free to use, adapt, or extend it for academic or professional use (with credit).
