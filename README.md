# Adversarial_attacks_and_robustness_for_ML_and_QML

**Comprehensive evaluation of adversarial robustness in classical ML and quantum-machine-learning classifiers**  

This repository contains a Jupyter Notebook that implements adversarial attack generation, robustness benchmarking, and interpretability analysis for both classical Machine Learning and Quantum Machine Learning models. The goal is to compare how classical neural networks and quantum variational circuits respond to gradient-based adversarial perturbations under the same experimental conditions.

---

## Table of Contents

- [Motivation & Objectives](#motivation--objectives)  
- [Experimental Setup](#experimental-setup)  
  - [Models](#models)  
  - [Attack Methods](#attack-methods)  
  - [Evaluation Strategy](#evaluation-strategy)  
- [Usage](#usage)  
  - [Requirements](#requirements)  
  - [Run Instructions](#run-instructions)  
- [Results & Insights](#results--insights)  
- [Limitations & Future Work](#limitations--future-work)  
- [License](#license)

---

## Motivation & Objectives

The rapid development of machine learning has enabled high-performance models on many tasks. However, classical ML models are known to be vulnerable to adversarial examples — very small input perturbations that can drastically change model outputs. The rise of Quantum Machine Learning (QML) brings new questions:

- Do QML variational classifiers exhibit the same vulnerability to adversarial attacks?  
- Under identical training regimes, how do classical ML and QML models compare in terms of robustness?  
- Can we use interpretability tools to analyze and compare the decision boundaries under adversarial perturbations?

This project aims to systematically address these questions by building both classical and quantum classifiers, subjecting them to standard adversarial attacks, and comparing their robustness and interpretability characteristics.

---

## Experimental Setup

### Models

- **Classical ML classifier**  
  - Implemented using PyTorch  
  - Typical feed-forward architecture with softmax output  
  - Trained with cross-entropy loss + Adam optimizer  

- **Quantum classifier (QML)**  
  - Built with PennyLane (or similar quantum-ML framework)  
  - Variational quantum circuit encoding classical input into qubit rotations  
  - Learnable variational parameters optimized via gradient descent  
  - Outputs class probabilities via measurement + softmax  

Both models are trained on the same dataset and under identical hyperparameter settings to ensure fair comparison.

### Attack Methods

- **Fast Gradient Sign Method (FGSM)** — single-step, sign-based perturbation of input along gradient direction.  
- **Projected Gradient Descent (PGD)** — iterative variant of FGSM, performing multiple small steps and clipping to ensure perturbation stays within ε-ball.  

These white-box gradient-based attacks probe the sensitivity of the models to adversarial perturbations.

### Evaluation Strategy

- Perform multiple independent training runs (e.g., 10 Monte Carlo runs) to account for randomness in initialization and training.  
- For each run, apply adversarial attacks with varying perturbation strengths (ε) and record classification accuracy degradation.  
- Visualize robustness curves (accuracy vs. ε), error bars across runs, and compare classical ML vs. QML behavior.  
- Use interpretability tools (saliency, integrated gradients, Grad-CAM, noise-smoothed maps) to visualize decision boundary sensitivity before and after attacks.

---

## Usage

### Requirements

- Python 3.10 or newer  
- (Recommended) GPU for faster training  
- Python packages:  

  pennylane
  torch
  torchvision
  numpy
  matplotlib
  captum

* (Optional) Additional dependencies depending on dataset or utility code

### Run Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/SiriusW823/Adversarial_attacks_and_robustness_for_ML_and_QML.git
   cd Adversarial_attacks_and_robustness_for_ML_and_QML
   ```
2. (Optional) Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate     # Linux / macOS  
   # or `venv\Scripts\activate` on Windows  
   pip install -r requirements.txt    # 如果你提供 requirements.txt  
   ```
3. Open the notebook:

   ```bash
   jupyter notebook Adversarial_attacks_and_robustness_for_ML_and_QML.ipynb
   ```
4. Execute all cells sequentially to run the entire pipeline: training, attack generation, evaluation, and visualization.

---

## Results & Insights

From the experimental pipeline, the following patterns emerge:

* Classical ML classifiers typically show higher vulnerability: small ε perturbations lead to substantial degradation in accuracy under FGSM/PGD.
* QML variational classifiers often exhibit smoother robustness curves: accuracy declines more slowly under identical attack budgets, suggesting a more stable gradient landscape.
* Interpretability maps (saliency, integrated gradients, Grad-CAM) reveal that classical models’ attribution maps distort more under adversarial perturbations, whereas QML models’ attribution remains comparatively coherent.

These preliminary findings indicate that QML classifiers may offer enhanced resilience to gradient-based adversarial attacks compared to their classical counterparts.

> ⚠️ **Note**: These observations are empirical. They do not constitute a proof of security or adversarial immunity — only comparative robustness under tested conditions.

---

## Limitations & Future Work

* The dataset used is relatively simple; real-world datasets may lead to different behavior.
* Only white-box, gradient-based attacks (FGSM / PGD) are considered; other adversarial threat models (e.g., black-box, universal perturbations) are not explored.
* No adversarial training or certified defense mechanisms are implemented.
* Quantum models are constrained by current QML frameworks; results may not generalize to larger circuits or real quantum hardware.
---

## License

This project is licensed under the MIT License.
See the LICENSE file for details.
