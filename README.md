# OpenAI One-Shot and Few-Shot Prompting Demo

A simple Python script to demonstrate and compare one-shot and few-shot prompting techniques for sentiment analysis using the OpenAI API.

---

## 🚀 Features

* **One-Shot Learning:** Provides the model with a single example to classify sentiment.
* **Few-Shot Learning:** Provides the model with multiple examples for more context-rich classification.
* **Secure API Key Handling:** Uses environment variables to protect your credentials.

---

## 📋 Requirements

* Python 3.8+
* An active OpenAI API key.

---

## 🛠️ Installation & Setup

Follow these steps to get the project running on your local machine.

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

**2. Install dependencies:**
It's recommended to use a virtual environment.
```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

**3. Set up your environment variables:**
Create a file named `.env` in the root of the project and add your OpenAI API key.

* First, copy the example file:
    ```bash
    cp .env.example .env
    ```
* Then, open the `.env` file and add your key:
    ```
    OPENAI_API_KEY="sk-..."
    ```
This `.env` file is ignored by Git and will not be uploaded.

---

## ▶️ How to Run

Execute the script from your terminal:

```bash
python your_script_name.py
```

You should see output similar to this:

```
One-shot Result: Negative
Few-shot Result: Mixed
```

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
