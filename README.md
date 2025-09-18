# Virtual-Keyboard

The Gesture-Controlled Virtual Keyboard is a real-time system that lets users search Google, YouTube, and Instagram using only hand gestures. Built with Python, OpenCV, and MediaPipe, it detects fingertip taps on a virtual keyboard and auto-executes queries, offering touchless, accessible, and intuitive interaction.

## Features

- **Real-time hand gesture recognition** using MediaPipe
- **Dynamic virtual keyboard** with multi-language support (English, Spanish, French, German)
- **Automatic search execution** on multiple platforms (Google, YouTube, Instagram)
- **Touchless interaction** for improved accessibility
- **Voice feedback** with text-to-speech
- **High contrast mode** for visual accessibility
- **Custom gesture training** and recognition
- **Mobile/tablet support** with Kivy interface

## Quick Start

### 1. Try the Demo (No Dependencies Required)
```bash
python3 demo.py
```
This interactive demo shows the virtual keyboard functionality without requiring a camera.

### 2. Full Installation
```bash
pip install -r requirements.txt
python3 main.py
```

### 3. Mobile Version
```bash
pip install kivy
python3 mobile_app.py
```

## Gesture Controls

- **Pinch (Thumb + Index)**: Type/Select keys
- **Swipe left/right**: Switch platforms
- **Hand open for 3s**: Toggle keyboard visibility
- **Two fingers up**: Toggle high contrast mode
- **Two fingers down**: Toggle voice feedback

## Keyboard Shortcuts

- **'q'**: Quit application
- **'t'**: Start gesture training mode
- **'s'**: Save trained gesture model
- **'c'**: Toggle high contrast mode
- **'v'**: Toggle voice feedback
- **'l'**: Switch language

## Supported Platforms

- **Google Search**: Web search queries
- **YouTube**: Video search
- **Instagram**: Hashtag exploration
- **Browser**: General web search

## Use Cases

- **Accessibility**: Hands-free computer interaction for motor disabilities
- **Public Kiosks**: Touchless interfaces in public spaces
- **Education**: Interactive learning and language practice
- **Creative Applications**: Gesture-controlled art installations and games
- **Medical Environments**: Sterile interaction systems

## Documentation

For detailed documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)

## Requirements

- Python 3.7+
- Webcam/Camera (for main application)
- Internet connection (for search and TTS)

## License

MIT License - see LICENSE file for details.