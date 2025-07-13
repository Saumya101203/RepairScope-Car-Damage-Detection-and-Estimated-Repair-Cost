# 🚗 RepairScope: Car Damage Detection and Repair Cost Estimator

**RepairScope** is a deep learning-powered web application that detects the type of damage in car images (e.g., front breakage, side crushed, rear normal) and estimates the potential repair cost based on both the detected damage and the selected car brand/model.

Built using **PyTorch**, **ResNet50**, and **Streamlit**, it delivers real-time predictions through an intuitive web interface. Ideal for users, insurance companies, or service centers needing fast, automated repair estimates.

---

## 🔍 Features

- 🔎 Detects damage type from uploaded car images across **8 custom classes**  
- 💰 Predicts estimated repair cost based on **damage severity** and **car brand/model**  
- 🖼️ Real-time predictions with confidence scores  
- 🌐 Deployed as an interactive Streamlit web app  
- 📦 Custom-trained on 2,664 annotated car images  
- 🔧 Hyperparameter tuning using **Optuna**  
- 📊 Clean UI with confidence bar plot & detailed result  

---

## 🧠 Model Architecture

Final model: **ResNet50 (transfer learning)**  
- Fine-tuned on 8 damage classes:  
  `['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal', 'Side Damaged', 'Side Normal']`  
- Data Augmentation: `torchvision.transforms`  
  (Random flip, rotation, color jitter, normalization)  
- Trained on: 2,131 images  
- Validated on: 533 images  

---

## 🚀 Model Performance

| Model                                 | Dropout | Learning Rate | Validation Accuracy | Notes                                  |
|--------------------------------------|:-------:|:--------------:|:-------------------:|----------------------------------------|
| **CNN (Baseline)**                   |   —     |     0.001      |       58.35%        | Simple CNN architecture                |
| **CNN + Regularization**             |   —     |     0.001      |       49.72%        | Batch Normalization + Weight Decay     |
| **EfficientNet (Transfer Learning)** |  0.3    |     0.001      |       67.92%        | Moderate boost with pretraining        |
| **ResNet50 (Transfer Learning)**     |  0.3    |     0.001      |       77.49%        | Good accuracy without tuning           |
| ✅ **ResNet50 + Optuna (Best)**      | 0.292   |     0.0024     |     **82.18%**      | Tuned via Optuna (final deployed model) |


---

## 🧪 Confusion Matrix

The model's performance across 8 damage categories is visualized below:

![Confusion Matrix](https://github.com/Saumya101203/RepairScope-Car-Damage-Detection-and-Estimated-Repair-Cost/blob/main/Code/confusion.png)

> The confusion matrix shows strong performance in identifying "Side Damaged" and "Front Normal" classes, with room for improvement in rear damage cases.

---

## 🌐 Live Demo

[🔗 Streamlit App (Click to View)](https://repairscope-car-damage-detection-and-estimated-repair-cost-pdb.streamlit.app/)  
<!-- Replace the above with your actual deployed link -->

---

## 🛠️ Tech Stack

| Layer       | Tools / Libraries                          |
|-------------|---------------------------------------------|
| Frontend    | Streamlit, Matplotlib (for visualizations) |
| Backend     | PyTorch, TorchVision, PIL                  |
| Training    | Optuna (for HPO), torchvision.transforms   |
| Model       | ResNet50 (pretrained), EfficientNet        |
| Deployment  | Streamlit Cloud                            |

---

## 📁 Project Structure

```bash
.
├── Streamlit_App/
│   ├── app.py                   # Streamlit frontend (UI + predictions)
│   ├── model/
│   │   └── saved_model.pth      # Final trained ResNet50 model (Optuna tuned)
│   ├── model_helper.py          # Inference and repair cost estimation logic
│   ├── requirements.txt         # Python dependencies for deployment
│   └── temp_file.jpg            # Temporary uploaded image file
│
├── Code/                        # Model training and experimentation notebooks
│   ├── damage_prediction.ipynb  # Contains model training pipeline
│   ├── hyperparameter_tunning.ipynb  # Optuna-based tuning workflow
│   └── saved_model.pth          # Final trained ResNet50 model (Optuna tuned)
```

---

## 📌 Notes

- 🔐 The training dataset is private and not shared due to licensing.  
- 📤 The `.pth` model file is tracked via Git LFS.  

---

## 📬 Contact

For queries or collaborations:  
📧 [saumya101203@gmail.com](mailto:saumya101203@gmail.com)

---







