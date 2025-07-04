
# 🎬 IMDB Sentiment Classification with Fine-tuned LLaMA

A complete fine-tuning pipeline for binary sentiment classification using TinyLlama-1.1B on the IMDB Movie Reviews dataset.

---

## 📋 Project Overview

This project demonstrates end-to-end fine-tuning of a large language model for sentiment analysis. Using **LoRA (Low-Rank Adaptation)** for efficient training, we adapt TinyLlama-1.1B to classify movie reviews as positive or negative.

> **Key Achievement :** 🏆 89% accuracy on IMDB sentiment classification

---

## 🚀 Quick Start

### 🔧 Prerequisites

```bash
pip install datasets transformers peft accelerate torch sklearn matplotlib seaborn
```

### ▶️ Run the Notebook

1. Open the notebook in **Google Colab**
2. Run the first cell to install dependencies
3. Execute all cells sequentially  
4. Total runtime: ~45–60 minutes

---

## 📊 Dataset

- **Source:** [IMDB Movie Reviews](https://huggingface.co/datasets/imdb)
- **Size:** 50,000 reviews (25k train, 25k test)
- **Classes:** Binary (Positive = 1, Negative = 0)
- **Preprocessing:** Balanced random sampling (15k train, 7.5k test)

---

## 🏗️ Architecture

### 🧠 Base Model

- **Model:** `TinyLlama-1.1B-Chat-v1.0`
- **Params:** 1.1B total, 1.1M trainable (0.1%)
- **Framework:** Hugging Face Transformers

### ⚙️ Fine-tuning Configuration

| Setting            | Value               |
|--------------------|---------------------|
| **Method**         | LoRA                |
| **Rank**           | 8                   |
| **Alpha**          | 16                  |
| **Dropout**        | 0.1                 |
| **Target Modules** | `q_proj`, `v_proj`  |

### 🏋️ Training Setup

- **Epochs:** 2
- **Effective Batch Size:** 16 (2 per device × 8 gradient accumulation)
- **Learning Rate:** 2e-4
- **Scheduler:** Cosine decay
- **Precision:** FP16

---

## 📈 Results

### 📉 Performance Metrics

| Metric                | Score |
|------------------------|-------|
| **Accuracy**           | 89%   |
| **Precision (Negative)** | 94%   |
| **Precision (Positive)** | 84%   |
| **Recall (Negative)**    | 85%   |
| **Recall (Positive)**    | 93%   |
| **F1-Score**             | 89%   |

### 📊 Confusion Matrix

```
                Predicted
           Negative  Positive
Actual Negative  46       8
       Positive   3      43
```

---

## 🔧 Key Features

### 🧪 Technical Highlights

- **Balanced Sampling:** Avoids class order bias
- **Efficient Fine-tuning:** 99.9% fewer trainable parameters via LoRA
- **Robust Evaluation:** Includes fallback keyword-based generation
- **Memory Optimized:** Gradient accumulation + FP16

### 🧹 Code Quality

- Reproducible with fixed seeds
- Extensive documentation
- Clear visualizations (training curves, confusion matrix)
- Fallbacks for error-prone evaluations

---

## 📁 Project Structure

```
├── IMDB_Sentiment_Classification.ipynb    # Main notebook
├── README.md                              # This file
├── requirements.txt                       # Dependencies
└── outputs/
    ├── training_curves.png               # Loss visualization
    ├── confusion_matrix.png              # Results visualization
    └── model_checkpoints/                # Saved LoRA adapters
```

---

## 🎯 Usage Examples

### 🔍 Basic Inference

```python
model = load_model_with_lora("./checkpoint-1800")

text = "This movie was absolutely fantastic!"
prediction = predict_sentiment(text, model, tokenizer)
# Output: "Positive"
```

### 📦 Batch Predictions

```python
reviews = [
    "Great movie, loved it!",
    "Terrible waste of time.",
    "The acting was decent but plot confusing."
]

predictions = [predict_sentiment(review, model, tokenizer) for review in reviews]
# Output: ["Positive", "Negative", "Negative"]
```

---

## 🧠 Technical Details

### 🧹 Data Preprocessing

- **Tokenizer:** TinyLlama tokenizer (max length = 512)
- **Input Format:** "Review: {text}\nSentiment: {label}"
- **Class Balance:** 50/50 via random sampling

### 🧱 Model Design

- Decoder-only transformer
- LoRA adapters added to attention layers
- Sentiment inferred through token probabilities

### 🧪 Evaluation Methods

- **Primary:** Token probability comparison (`Positive` vs `Negative`)
- **Fallback:** Generated text sentiment keyword analysis
- **Metrics:** Full classification report

---

## 🚨 Important Notes

- **Data Bias:** Original IMDB dataset is sorted by label — careful sampling was critical
- **Compute:** Minimum 12GB VRAM recommended
- **Runtime:** 45–60 mins on Colab
- **Subset Used:** 22.5k samples (to speed up fine-tuning and reduce memory)

---

## 🤝 Contributing

We welcome contributions! Try:
- New base models or LoRA configs
- Improved token evaluation techniques
- Alternative sampling strategies
- More visualizations or metrics

---

## 📚 References

- [TinyLlama on GitHub](https://github.com/jzhang38/TinyLlama)
- [LoRA: Low-Rank Adaptation Paper](https://arxiv.org/abs/2106.09685)
- [IMDB Dataset on HF](https://huggingface.co/datasets/imdb)

---

## 📄 License

This project is licensed under the **MIT License**.

---

**Built with:** 🤗 Transformers • 🔥 PyTorch • 📊 scikit-learn • 🎨 Matplotlib • 🐍 Python

*For questions or issues, please open an issue in the repository.*
