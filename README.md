# Crowd Density Tracking System

A comprehensive AI-powered system for real-time crowd density analysis, prediction, and alerting in metro/train environments. This project combines YOLOv8 object detection with optical flow algorithms to accurately track crowd levels, forecast overcrowding, and provide actionable insights through interactive dashboards.

## 🚀 Features

- **Dual Detection Algorithms**:
  - YOLOv8 for sparse to medium density crowds (individual person detection)
  - Optical Flow (IEEE 7275777 methodology) for dense, occluded crowds
- **Real-time Crowd Forecasting**: Predicts future crowd levels using LSTM models and exponential smoothing
- **Intelligent Alert System**: Multi-channel notifications (console, email, SMS, Slack) with configurable thresholds
- **Interactive Dashboard**: HTML-based dashboard with real-time visualizations, risk assessments, and operator recommendations
- **Batch Processing**: Automated processing of multiple video feeds with comprehensive reporting
- **Automatic Learning**: System improves predictions over time by retraining on new data
- **Zone-based Analysis**: Configurable zones for targeted monitoring and counting
- **Data Management**: Automatic cleanup of old/redundant data to maintain efficiency

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for YOLO processing)
- Windows/Linux/macOS

## 🛠 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/crowd-density-tracking.git
   cd crowd-density-tracking
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\activate
   # On Unix/MacOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO models:**
   - The repository includes `yolov8n.pt` and `yolov8s.pt` models
   - For best performance, ensure models are in the project root

## 🎯 Quick Start

### Batch Processing (Recommended)

Process all videos with predictions and generate reports:

```bash
python batch_processor.py --headless
```

### Real-time Analysis

Analyze a single video file:

```bash
python main.py --source path/to/video.mp4 --algo yolo
```

Use optical flow for dense crowds:

```bash
python main.py --source path/to/video.mp4 --algo flow
```

## 📖 Usage Guide

### Command Line Options

#### Main Script (`main.py`)

- `--source`: Video file path or camera index (default: None)
- `--model`: Path to YOLO model (default: yolov8s.pt)
- `--conf`: Confidence threshold for detection (default: from config)
- `--algo`: Detection algorithm - 'yolo' or 'flow' (default: yolo)
- `--delay`: Delay between frames in seconds (default: 1)

#### Batch Processor (`batch_processor.py`)

- `--headless`: Run without GUI (default: False)
- `--limit`: Limit number of videos to process (default: all)
- `--loop`: Continuous processing loop (default: False)
- `--algo`: Algorithm to use (default: yolo)
- `--alerts`: Alert channels - 'console', 'email', 'sms', 'slack' (can specify multiple)

### Configuration

Edit `config/config.py` to customize:

- Detection thresholds
- Alert levels
- Zone definitions
- Forecasting parameters

## 🏗 Architecture

The system employs a hybrid architecture:

1. **Sparse Pipeline (YOLOv8)**: Individual person detection for low-medium density crowds
2. **Dense Pipeline (Optical Flow)**: Pixel-level motion tracking for occluded, high-density scenarios
3. **Forecasting Engine**: LSTM-based prediction models for crowd level forecasting
4. **Alert Engine**: Configurable threshold-based alerting system
5. **Visualization Layer**: Real-time HUD overlays and interactive dashboards

### Key Components

- `utils/detector.py`: YOLOv8 integration with ByteTrack
- `utils/optical_flow_pipeline.py`: Dense crowd analysis using Farneback flow
- `utils/forecaster.py`: Time-series predictions and risk assessment
- `utils/dashboard.py`: HTML report generation
- `forecasting/lstm_model.py`: Neural network for crowd forecasting

## 📊 Outputs

After processing, the system generates:

- **`dashboard.html`**: Interactive web dashboard with visualizations
- **`batch_report_TIMESTAMP.csv`**: Detailed frame-by-frame data
- **`batch_summary_TIMESTAMP.txt`**: Human-readable summary report
- **`alerts_TIMESTAMP.csv`**: Alert log with timestamps and severity
- **`report_TIMESTAMP.json`**: Machine-readable data export

## 🔧 Development

### Project Structure

```
Crowd-density-tracking-main/
├── main.py                    # CLI entry point
├── batch_processor.py         # Batch processing engine
├── requirements.txt           # Python dependencies
├── config/                    # Configuration files
│   ├── config.py
│   ├── zones.py
│   └── alerts_config.json
├── utils/                     # Core utilities
│   ├── detector.py
│   ├── optical_flow.py
│   ├── forecaster.py
│   └── ...
├── forecasting/               # ML forecasting models
├── data/dataset/              # Video data directory
└── outputs/                   # Generated reports and dashboards
```

### Adding New Features

1. Implement new utilities in the `utils/` directory
2. Update configuration in `config/config.py`
3. Add CLI options to `main.py` argument parser
4. Update this README with new usage instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics for object detection
- OpenCV for computer vision algorithms
- PyTorch for deep learning framework
- IEEE 7275777 methodology for optical flow crowd analysis

## 📞 Support

For questions or issues:

- Open an issue on GitHub
- Check the [QUICKSTART.md](QUICKSTART.md) for common usage
- Refer to [PREDICTION_GUIDE.md](PREDICTION_GUIDE.md) for detailed system documentation</content>
  <parameter name="filePath">d:\metro crowding\Crowd-density-tracking-main\README.md
