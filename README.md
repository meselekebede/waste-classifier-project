# 🌟 Image-Based Waste Classification using ResNet-34
This project implements an image-based waste classification system using the ResNet-34 architecture with transfer learning. The model classifies waste into six categories:

Cardboard
Glass
Metal
Paper
Plastic
Trash
The model was trained using PyTorch and is now deployed as a web app using Streamlit , allowing users to upload images and receive predictions in real-time.

🎯 Test Accuracy : ~83%
⏱️ Training Time : ~200 minutes
🧠 Model Type : ResNet-34 (Transfer Learning)

## 📷 Try It Out!
https://waste-classifier-project.streamlit.app/

Upload an image of waste, and the AI will tell you what type it thinks it is — along with confidence scores for each category.

# 🧠 How It Works
Uses pre-trained ResNet-34 weights from PyTorch
Replaces final layer to classify 6 types of waste
Applies data normalization and mixed precision training for efficiency
Implements early stopping and learning rate scheduling
Evaluates using accuracy, F1-score, precision, and recall
#📁 Dataset
The dataset used is sourced from the TrashNet dataset by Gary Thung and Martin Mangino:

https://github.com/garythung/trashnet

Folder structure expected:

dataset/

    cardboard/
    glass/
    metal/
    paper/
    plastic/
    trash/
## 🛠️ Technologies Used: 
PyTorch: Deep learning framework

ResNet-34: Pre-trained CNN model

Streamlit: Web interface for deployment

TorchVision: Image transforms and models

scikit-learn: Evaluation metrics (F1, Precision, Recall)

## 🚀 Local Setup
Clone the repo

git clone https://github.com/meselekebede/waste-classifier-project.git 


cd waste-classifier

## Install dependencies,
Make sure you're using Python 3.9+


pip install -r requirements.txt


## Train or Run the App


To train the model:

python resnet_train.py

To run the web app:

streamlit run app.py


## 📦 Requirements

torch==1.13.1

torchvision==0.14.1

streamlit==1.24.0

Pillow==9.5.0

matplotlib==3.7.2

scikit-learn==1.3.0

tqdm

numpy

# 📊 Model Performance
Test Accuracy: 83%

F1 Score: 0.83

Precision: 0.83

Recall: 0.83

Training Time: ~200 min

# 📝 Notes
This is an early-stage prototype and may misclassify some items.
Designed for educational purposes and as a foundation for smart waste management systems.
Easily extendable to support more classes like e-waste, biodegradable, etc.
# 🤝 Contributing

Contributions are welcome! If you'd like to:

Add new features (e.g., more classes),
Improve accuracy with data augmentation,
Convert to mobile app or backend API,
Feel free to open an issue or submit a pull request.

# 📬 Feedback & Questions
If you have any questions or want help deploying this yourself, feel free to reach out or leave a comment!

# 🏆 Credits

Dataset: [TrashNet](https://github.com/garythung/trashnet)

Frameworks: PyTorch, TorchVision, Streamlit
