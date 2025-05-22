# Enhanced GAT Model for Node Classification on Cora Dataset

This project showcases an improved implementation of a **Graph Attention Network (GAT)** for **node classification** on the **Cora citation dataset**, replacing the original **Graph Convolutional Network (GCN)** baseline that was provided by the instructor.

The goal was to go beyond the fixed-weight aggregation of GCN by introducing a dynamic, attention-based architecture with advanced regularization techniques to achieve higher accuracy, better generalization, and improved training stability.

---

## Technologies Used

| Component       | Configuration                  |
|----------------|----------------------------------|
| Dataset         | Cora (via PyTorch Geometric)    |
| Framework       | PyTorch + PyTorch Geometric     |
| Visualization   | matplotlib, seaborn             |
| Optimizer       | Adam                            |
| Training Device | GPU (if available)              |

---

## Project Overview

- **Original Setup:** Instructor-provided GCN implementation  
- **Improved Version:** Custom 2-layer GAT with 8 attention heads  
- **Enhancements Added:**
  - `LayerNorm` for training stability  
  - `DropEdge` for structural regularization  
  - `Gradient Clipping` to prevent exploding gradients  
- **Task:** Semi-supervised node classification  
- **Dataset:** Cora (2,708 nodes, 10,556 edges, 7 classes)

---

## ✅ Results

| Metric        | Value         |
|---------------|---------------|
| Validation Accuracy | ~81.8%     |
| Test Accuracy       | **81.4%**  |
| Best Epoch          | 183        |

- Replacing GCN with GAT resulted in **significant accuracy improvement**  
- Regularization (DropEdge, LayerNorm) helped prevent overfitting  
- Model showed **stable convergence and better generalization**
