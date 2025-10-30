#  BERT Sentiment Pipeline

A collaborative end-to-end **Sentiment Analysis** project built by **Nsangou**, **Gabi**, and **Teddy**.  
It simulates a BERT-style machine-learning workflow — from data extraction to inference — with Continuous Integration (CI) using GitHub Actions.

---

##  Project Overview
This project demonstrates how a small team can build a complete NLP pipeline while applying MLOps principles such as:
- **Version control & branching** (Git / PR workflow)
- **Automated testing** (Pytest)
- **Continuous Integration** (GitHub Actions)
- **Collaborative management** (Trello board)

---

## Setup Instructions

### Clone and set up
```bash
git clone https://github.com/Nsangou22/bert-sentiment-pipeline.git
cd bert-sentiment-pipeline
python -m venv venv
venv\Scripts\activate   # on Windows
pip install -r requirement.txt
```

---

##  Components Overview

### `data_extraction.py`
Loads and validates the dataset (CSV).  
Handles missing files or invalid columns gracefully.

###  `data_processing.py`
Cleans text (removes punctuation, URLs, emojis), tokenizes into IDs, and splits data into training/testing sets.

###  `model.py`
Implements a lightweight **DummyModel** that mimics a BERT classifier:
- `forward()` → returns random logits `[batch_size, num_labels]`
- `predict()` → converts text into logits  
Used to simulate fine-tuning and inference behavior.

### 🔍 `inference.py`
Uses the model to classify new text as **Positive** or **Negative**.  
This represents the final stage of the pipeline.

---

##  Testing & Continuous Integration

**Tool:** Pytest  
**Location:** `/tests/unit/`

- Each component has its own test file.  
- Total: 12 unit tests (all passed).  
- Tests run automatically through **GitHub Actions**.

**Workflow file:** `.github/workflows/ci.yml`  
Every push or pull request triggers the CI job to:
1. Install dependencies  
2. Run pytest  
3. Display “ All checks passed” before merging  

---

## Collaboration & Workflow

**Branching model:**  
`feature/data-extraction` → `feature/data-processing` → `feature/model-training` → `feature/inference`  

Each branch had its own pull request (PR) and was reviewed before merging into `main`.  
All teamwork was tracked through **Trello** for tasks and progress.

---

##  Results Summary

| Phase | Description | Tests | Status |
|-------|--------------|--------|--------|
| Data Extraction | Dataset loading & validation | 3 | ✅ |
| Data Processing | Cleaning, tokenizing, splitting | 4 | ✅ |
| Model | Dummy BERT simulation | 3 | ✅ |
| Inference | Sentiment prediction | 2 | ✅ |
| **Total** |  | **12** | 🟢 All Passed |

---

## Team Roles

| Member | Focus |
|--------|--------|
| **Abdel** | Project setup, data extraction |
| **Teddy** | Data processing, model |
| **Gabi** | CI/CD setup, inference & code review |

---

## Conclusion

This project reproduces a realistic MLOps workflow:
- Modular, testable Python code  
- Automated testing with CI/CD  
- Collaborative Git branching and code review  
- End-to-end sentiment prediction pipeline  


> *Built with teamwork, automation, and clean design.*
