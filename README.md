# Sentiment Analysis: A Comparative Study of ML and DL Models

This project provides a comprehensive comparison between classical Machine Learning models and various Deep Learning architectures for sentiment analysis. The analysis is performed on the well-known IMDB Movie Review dataset to classify reviews as either positive or negative.

---

## 📈 Analysis Overview

The notebook is divided into two main parts, followed by a state-of-the-art model fine-tuning:

* **Part 1: Classical Machine Learning:**
    * Text is preprocessed using **spaCy** for lemmatization and removal of stop words/punctuation.
    * Features are extracted using **TF-IDF**.
    * Four different models are trained and evaluated: **SVM**, **K-Nearest Neighbors**, **Decision Tree**, and **Random Forest**.

* **Part 2: Deep Learning with PyTorch:**
    * A custom vocabulary is built, and reviews are tokenized and padded.
    * Three deep learning architectures are implemented:
        * **Convolutional Neural Network (CNN)**
        * **Long Short-Term Memory (LSTM)**
        * **Gated Recurrent Unit (GRU)**

* **Part 3: Transformer Model Fine-Tuning:**
    * A pre-trained **BERT (bert-base-uncased)** model from Hugging Face is fine-tuned on the sentiment analysis task, demonstrating the power of transfer learning.

---

## 📊 Performance Highlights

The fine-tuned BERT model achieved the highest accuracy, slightly outperforming the best classical model (SVM). The custom deep learning models (GRU, LSTM, CNN) performed well but were not as accurate as the larger models, though they trained significantly faster.

| Model Category      | Model         | Accuracy | Training Time (approx.) |
| ------------------- | ------------- | :------: | :---------------------: |
| **Transformer (DL)**| **BERT** | **89.3%**|        ~28 minutes        |
| **Classical ML** | SVM           |  88.5%   |        ~27 minutes        |
| **Recurrent NN (DL)** | GRU           |  86.1%   |         ~11 seconds         |

---

## 🛠️ Technologies & Libraries Used

* **Python 3**
* **Data Manipulation:** Pandas, NumPy
* **NLP Preprocessing:** spaCy
* **Machine Learning:** Scikit-learn (`TfidfVectorizer`, `SVC`, `KNeighborsClassifier`, etc.)
* **Deep Learning:** PyTorch (`nn`, `optim`, `DataLoader`)
* **Transformers:** Hugging Face `transformers` (for BERT)
* **Environment:** Jupyter Notebook

---

## 💿 Dataset

This project uses the **IMDB Dataset of 50K Movie Reviews**. It's a binary sentiment classification dataset containing 25,000 highly polar movie reviews for training and 25,000 for testing.

---

## 🚀 How to Run

To replicate this analysis on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install the required libraries:**
    It's recommended to use a virtual environment.
    ```bash
    pip install pandas spacy scikit-learn torch transformers
    ```
    *Note: You may need to install a specific version of PyTorch that matches your CUDA version if you plan to use a GPU.*

3.  **Download the spaCy model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

4.  **Get the dataset:**
    Download the `IMDB Dataset.csv` file (e.g., from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)) and place it in the root directory of the project.

5.  **Launch the notebook:**
    Open and run the `FINALsentiment_all.ipynb` notebook in a Jupyter environment like Jupyter Lab or Google Colab.
