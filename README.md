# Road Damage Detection using YOLO

Detection and localization of road damage from images using YOLO object detection,
with a Streamlit web app for image-based inference.

## Dataset
- **Source:** [road-damage Computer Vision Dataset (Roboflow Universe)](https://universe.roboflow.com/trafficsignssafeway/road-damage-l1ju7)
- **License:** CC BY 4.0
- **Size:** 6,757 images (Train 4,930 / Valid 1,146 / Test 681)
- **Classes (7):** alligator, block, crack, edge, longitudinal, pothole, transverse
- **Note:** The dataset is highly imbalanced (pothole and alligator dominate; `edge` has only 21 training examples and none in the test set).

## Training Notebook
Kaggle notebook (training, tuning, and evaluation): [road-damage on Kaggle](https://www.kaggle.com/code/razan3ssam/road-damage)
Also included in this repo: `notebook/road_damage.ipynb`

## Models Compared
Trained on Tesla T4 (Kaggle), 50 epochs each, same data and settings (AdamW, lr0=0.001, imgsz=640, batch=16, seed=0).

| Model | mAP50 | mAP50-95 | Precision | Recall | FPS | Size |
|---|---|---|---|---|---|---|
| YOLOv8n | 0.576 | 0.352 | 0.555 | 0.574 | 70.6 | 6.5 MB |
| **YOLO11n (selected)** | **0.590** | **0.372** | **0.636** | 0.537 | 61.6 | 5.5 MB |
| YOLO11n (SGD, lr0=0.01) | 0.588 | 0.362 | 0.592 | 0.560 | 61.2 | 5.5 MB |
| YOLO11n (imgsz=800) | 0.582 | 0.350 | 0.584 | 0.551 | 56.6 | 5.5 MB |

All metrics are on the **test set** (681 images). FPS is end-to-end on a T4 GPU.
Hyperparameter tuning did not improve over the default settings.

## Run the App
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open `http://localhost:8501`, upload a road image, and adjust the confidence threshold.

## Project Structure
- `app.py` — Streamlit app
- `v11n_best.pt` — trained YOLO11n weights
- `notebook/` — training and evaluation notebook
- `results/` — final comparison table

## Author
Razan Essam