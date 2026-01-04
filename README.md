# ☀️ Solar Panel Damage Detection System

An AI-powered solution for detecting damage and performance issues in solar panels using deep learning and computer vision. This system enables preventive maintenance, maximizes clean energy output, and reduces operational costs.

## 🎯 Overview

This project addresses the critical challenge of maintaining solar panel efficiency through early damage detection. By analyzing images from regular cameras, mobile phones, or drones, the system identifies:

- **Clean/Healthy panels** - Optimal condition
- **Soiling** - Dust, dirt accumulation
- **Bird droppings** - Biological contamination
- **Physical damage** - Cracks, breakage
- **Electrical damage** - Hotspots, connection issues
- **Snow coverage** - Weather-related obstruction

### Key Benefits

- 🔍 **Early Detection**: Identify issues before they cause major energy loss
- 💰 **Cost Savings**: Estimated potential maintenance cost reduction (~30%) based on preventive maintenance assumptions
- 🌍 **Environmental Impact**: Maximize clean energy output, reduce e-waste
- ⚡ **Efficiency**: Potential 3-5 year lifespan extension based on early detection (industry literature)
- 📱 **Accessibility**: Works with any camera (phone, drone, CCTV)

## 🚀 Features

### Layer 1: Core AI Engine
- ✅ 6-class damage classification
- ✅ Transfer learning (MobileNetV2/EfficientNet)
- ✅ Confidence scoring for predictions
- ✅ Camera-agnostic design

### Layer 2: Impact Analysis
- 📊 Energy loss estimation
- 🌍 CO₂ impact calculation
- 💵 Financial impact assessment
- ⚠️ Maintenance priority scoring

### Layer 3: Explainability
- 🔍 Grad-CAM visual explanations
- 📈 Class probability distributions
- 💡 Actionable recommendations
- 📋 Decision transparency

## 📋 Prerequisites

- Python 3.8+
- pip package manager
- (Optional) Kaggle API credentials for dataset download
- 4GB+ RAM recommended
- GPU recommended for training (CPU works but slower)

## 🔧 Installation

### 1. Clone or Download

```bash
# If you have the file
cd /path/to/project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Kaggle API (For Training)

To download training datasets from Kaggle:

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Save `kaggle.json` to `~/.kaggle/`
5. Set permissions:

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 🎓 Usage

### Web Interface (Recommended)

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Training (Command Line)

The training pipeline is separated from the web app for better organization.

#### Download Datasets
```bash
python train.py --download-data
```

#### Train Model
```bash
# Using MobileNetV2 (faster, lighter)
python train.py --train --epochs 50 --architecture mobilenet

# Using EfficientNetB0 (more accurate)
python train.py --train --epochs 50 --architecture efficientnet
```

#### Fine-tune Existing Model
```bash
python train.py --fine-tune --epochs 20
```

#### Evaluate Model
```bash
python train.py --evaluate
```

## 📊 Dataset Information

The system uses two high-quality Kaggle datasets:

1. **Solar Panel Clean and Faulty Images**
   - Dataset ID: `pythonafroz/solar-panel-images-clean-and-faulty-images`
   - Classes: Clean, Dusty, Bird-drop, Electrical-damage, Physical-damage, Snow

2. **PV Panel Defect Dataset**
   - Dataset ID: `alicjalenarczyk/pv-panel-defect-dataset`
   - 6 defect classes with balanced, curated images

Combined, these datasets provide comprehensive coverage of real-world solar panel conditions.

## 🏗️ System Architecture

```
Input Image (Any Camera)
    ↓
Image Preprocessing (224x224, Normalization)
    ↓
MobileNetV2/EfficientNet (Transfer Learning)
    ↓
Custom Classification Head
    ↓
6-Class Prediction + Confidence
    ↓
Impact Analysis Module
    ↓
Output: Damage Class, Priority, Energy Loss, CO₂ Impact, Recommendations
```

### Model Details

- **Base Architecture**: MobileNetV2 (ImageNet pre-trained) or EfficientNetB0
- **Input Size**: 224x224x3 RGB
- **Output**: 6 classes with softmax probabilities
- **Model Size**: ~15MB (MobileNet) / ~20MB (EfficientNet)
- **Inference Time**: <100ms on CPU, <20ms on GPU
- **Accuracy**: ~90% validation accuracy on curated Kaggle datasets; performance varies by class and data quality

### Reproducibility

**All results are reproducible using the provided training pipeline and configuration files.** The `train.py` script with default settings in `config.py` will produce consistent results across runs with the same datasets. Training logs, model checkpoints, and evaluation metrics are automatically saved for verification.

## 📱 Web Interface Guide

### Detection Tab
1. Upload solar panel image (JPG/PNG)
2. View instant analysis results
3. Check confidence scores and priority level
4. Review impact metrics (energy, CO₂, financial)
5. See Grad-CAM visual explanation
6. Get actionable recommendations

### Training Dashboard
1. Setup Kaggle API credentials
2. Download datasets with one click
3. Configure training parameters
4. Monitor training progress
5. View accuracy/loss curves

### Impact Analysis
1. Configure system parameters (# panels, capacity)
2. View system-wide projections
3. Estimate annual energy savings
4. Calculate CO₂ reduction potential

## 🎯 Key Metrics & Impact

### Performance Metrics
- **Energy Loss Detection**: 5-100% loss estimation by damage type
- **Maintenance Priority**: Low/Medium/High/Critical classification
- **Confidence Scoring**: 0-100% prediction reliability

### Environmental Impact
- **Energy Recovery**: Identifies 5-50% efficiency losses
- **CO₂ Reduction**: ~0.5 kg CO₂ per kWh saved (varies by grid mix)
- **Panel Lifespan**: Potential +3-5 year extension through early detection (based on industry literature)
- **E-Waste Reduction**: Estimated ~15% fewer premature replacements
- **Maintenance Cost**: Estimated potential ~30% reduction through preventive maintenance approach

### Financial Impact
- **Maintenance Cost**: Estimated potential ~30% reduction through preventive maintenance approach
- **Energy Savings**: Potential to recover 10-40% of lost production
- **ROI**: Indicative ROI range (6-18 months) for large installations (>100 panels), dependent on site conditions, local costs, and electricity rates

## 🔍 Example Results

### Clean Panel
```
Prediction: Clean
Confidence: 97.3%
Priority: Low
Energy Loss: 0%
Action: Continue routine monitoring
```

### Dusty Panel
```
Prediction: Dusty/Soiling
Confidence: 94.1%
Priority: Low
Energy Loss: 8-12%
Annual Impact: ~150 kWh lost
Action: Schedule cleaning within 2 weeks
```

### Physical Damage
```
Prediction: Physical-Damage
Confidence: 96.8%
Priority: High
Energy Loss: 25-35%
Annual Impact: ~500 kWh lost, $60 cost
Action: Immediate professional inspection
```

## 🛣️ Roadmap

### Phase 2 (Q2 2024)
- [ ] Thermal imaging integration
- [ ] Time-series damage progression tracking
- [ ] Mobile app (iOS/Android)
- [ ] Automated periodic monitoring

### Phase 3 (Q3 2024)
- [ ] Drone automation support
- [ ] IoT sensor fusion
- [ ] Predictive maintenance algorithms
- [ ] Fleet management dashboard

### Phase 4 (Q4 2024)
- [ ] ERP/CMMS integration
- [ ] Automated maintenance ticketing
- [ ] Performance analytics suite
- [ ] ROI tracking dashboard

## 🤝 Responsible AI

This system is designed as a **decision support tool** for maintenance professionals, not as a fully autonomous solution.

### Guidelines
- ✅ All predictions should be verified by qualified personnel
- ✅ Critical damage classifications require professional inspection
- ✅ System provides transparent, explainable predictions
- ✅ Confidence scores indicate prediction reliability
- ✅ Recommendations are advisory, not prescriptive

### Limitations
- Cannot detect all forms of degradation (e.g., internal cell issues)
- Requires clear, well-lit images for best results
- Performance may vary with extreme weather conditions
- Should be used alongside regular professional inspections

## 📊 Technical Specifications

### Supported Input Sources
- 📱 Mobile phone cameras (iOS/Android)
- 📷 Digital cameras (DSLR, point-and-shoot)
- 🚁 Drone cameras (DJI, commercial)
- 📹 CCTV/surveillance cameras
- 🖥️ Webcams

### System Requirements

#### Minimum (Inference Only)
- CPU: 2+ cores, 2.0+ GHz
- RAM: 4GB
- Storage: 500MB
- OS: Windows 10, macOS 10.14+, Linux (Ubuntu 18.04+)

#### Recommended (Training + Inference)
- CPU: 4+ cores or GPU (NVIDIA RTX series)
- RAM: 8GB+
- Storage: 5GB+ (for datasets)
- OS: Linux (Ubuntu 20.04+) or Windows 10/11

### Deployment Options
- ☁️ Cloud: AWS, GCP, Azure
- 🖥️ On-premise: Local servers
- 📱 Edge: Raspberry Pi 4, NVIDIA Jetson
- 🌐 Web: Streamlit Cloud, Heroku

## 🐛 Troubleshooting

### "Model not found" error
- Ensure you've trained the model first using the Training Dashboard
- Check that `models/solar_panel_model.h5` exists
- Try training with: `python solar_panel_detection.py --train`

### Low prediction confidence
- Ensure image is clear and well-lit
- Zoom in on the specific panel
- Avoid extreme angles or reflections
- Try different images of the same panel

### Kaggle API errors
- Verify `kaggle.json` is in `~/.kaggle/`
- Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
- Ensure you've accepted dataset terms on Kaggle website

### Training takes too long
- Reduce epochs: `--epochs 30`
- Use smaller batch size (edit CONFIG in code)
- Consider using GPU or cloud training
- Try MobileNet instead of EfficientNet

## 📄 License

This project is provided for educational and research purposes. Please check individual dataset licenses:
- Solar Panel Clean/Faulty Images: Check Kaggle dataset page
- PV Panel Defect Dataset: CC BY-NC-SA 4.0

## 🙏 Acknowledgments

- Kaggle datasets: pythonafroz, alicjalenarczyk
- TensorFlow/Keras teams for deep learning framework
- Streamlit for rapid web app development
- Transfer learning: ImageNet pre-trained models

## 📧 Support

For issues, questions, or contributions:
1. Check troubleshooting section above
2. Review code comments for implementation details
3. Test with different images/settings

## 🌟 Key Highlights

✨ **Modular Architecture**: Training, inference, and deployment components designed for easy integration
🚀 **Production-Ready**: Separated training pipeline and inference app for clean deployment
🎯 **Climate-Focused**: Designed to maximize renewable energy impact
📊 **Explainable AI**: Grad-CAM visualizations and transparent predictions
⚡ **Edge-Ready**: Lightweight model suitable for edge deployment
📱 **Accessible**: Works with any camera, no specialized hardware needed

---

**Mission**: Maximizing clean energy output through AI-powered preventive maintenance, reducing waste, extending lifespans, and optimizing renewable energy infrastructure. 🌞🌍
