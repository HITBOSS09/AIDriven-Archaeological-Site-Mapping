# 🌍 AI-Driven Archaeological Site Mapping

Detecting hidden archaeological sites using vegetation anomalies and soil pattern analysis through deep learning.

An AI-powered system that analyzes satellite and drone imagery to identify potential archaeological locations by studying terrain patterns, vegetation density, and soil characteristics.

🔗 **Live Demo**: https://archaeological-frontend-one.vercel.app/

---

## 🎥 Demo

![Segmentation Demo](assets/demo.gif)

*Segmentation output highlighting vegetation, soil regions, and potential archaeological zones.*

---

## 📊 Results

![Performance](assets/performance.png)

*The model segments imagery into vegetation and soil regions. Variations in vegetation density and soil patterns are used to identify potential archaeological sites.*

**Evaluation Metrics:**
- IoU (Intersection over Union)
- Dice Coefficient
- Accuracy

---

## 🧠 System Architecture

![Architecture](assets/architecture.png)

*Pipeline showing data flow from satellite imagery to prediction and visualization.*

---

## 🔬 Methodology

1. **Data Collection**
   - Satellite imagery / drone-based aerial data  

2. **Preprocessing**
   - Image resizing and normalization  
   - Noise reduction  
   - Data augmentation  

3. **Segmentation & Classification**
   - U-Net architecture for semantic segmentation  
   - Separation of vegetation and soil regions  
   - Terrain classification  

4. **Analysis**
   - Vegetation density estimation  
   - Soil pattern classification  
   - Detection of anomalies indicating buried structures  

5. **Prediction & Visualization**
   - Mask generation  
   - Overlay results on original images  

---

## 🚀 Features

- Automated archaeological site detection  
- Semantic segmentation using deep learning  
- Vegetation density analysis  
- Soil classification for terrain understanding  
- Detection of hidden patterns in terrain  
- Scalable pipeline for large-area satellite analysis  

---

## 🌍 Applications

- 🏛️ Archaeological site discovery  
- 🌱 Vegetation-based anomaly detection  
- 🧱 Soil classification for terrain mapping  
- 🛰️ Remote sensing & geospatial intelligence  
- 📍 Heritage preservation  

---

## ⚠️ Limitations

- Performance depends on image quality and resolution  
- Dense vegetation may obscure underlying structures  
- Requires annotated datasets for accurate training  

---

## 🔮 Future Work

- Integration with YOLO for object detection  
- Multi-spectral & thermal image support  
- GIS (Geographic Information Systems) integration  
- Real-time mapping dashboard  
- Drone-based automated surveying  

---

## 🛠️ Tech Stack

- Python  
- OpenCV  
- PyTorch / TensorFlow  
- U-Net (Semantic Segmentation)  
- NumPy, Matplotlib  

---

## 📂 Project Structure
