#End-to-End OCR-Free Image Translation for Low-Resource Languages

🚀 Implementation of [*Improving End-to-End Text Image Translation from the Auxiliary Text Translation Task*](https://arxiv.org/abs/2210.03887), adapted for **Gujarati** using a **Vision Transformer (ViT)** image encoder.  
This project demonstrates that we can translate images containing text **without relying on OCR**, especially in low-resource settings where OCR data is scarce or noisy.  

---

## ✨ Key Highlights
- 🔤 **Low-Resource Translation**: Extended the model to Gujarati, a language with limited OCR and parallel resources.  
- 🖼️ **Image Encoder Upgrade**: Replaced ResNet with **ViT** for better visual-text alignment.  
- 🏋️ **Hybrid Training Data**: Combined **100k real OCR-English pairs** with synthetic image–text data. 
- 🎯 **Impact**: Enables direct translation from images in languages with poor OCR availability, reducing error propagation from traditional OCR+MT cascades.  

---

## 📂 Dataset
- **Real Data**: 100,000 Gujarati text images with English translations.  
- **Synthetic Data**: Generated text images with diverse backgrounds and fonts to improve generalization.  
- **OCR Augmentation**: Used OCR-extracted Gujarati text aligned with English for supervised training.  
---

## 🏗️ Model Architecture
- **Image Encoder**: CLIP Vision Encoder.  
- **Text Encoder/Decoder**: Transformer-based sequence-to-sequence architecture.  
- **Training Strategy**:
  - Multi-task learning with auxiliary **MT** (text-to-text) and **OCR** tasks.  
  - Shared encoder-decoder parameters for alignment between text and image modalities.  

---

@article{ma2022improving,
  title={Improving End-to-End Text Image Translation From the Auxiliary Text Translation Task},
  author={Ma, Cong and Zhang, Yaping and Tu, Mei and Han, Xu and Wu, Linghui and Zhao, Yang and Zhou, Yu},
  journal={arXiv preprint arXiv:2210.03887},
  year={2022}
}
