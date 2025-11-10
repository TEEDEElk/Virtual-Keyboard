# 🎯 Installation & Setup Guide

## Prerequisites

- Python 3.8 or higher
- Webcam
- Modern web browser (Chrome/Edge for extension)

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/TEEDEElk/Virtual-Keyboard.git
cd Virtual-Keyboard
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application

#### Desktop Application
```bash
python main.py
```

#### Mobile Application (Kivy)
```bash
python mobile_app.py
```

Or build with Buildozer for Android:
```bash
buildozer android debug
```

## 🎮 Controls & Gestures

### Basic Gestures
- **Pinch (thumb + index finger)** - Type/Select keys
- **Open hand for 3 seconds** - Toggle virtual keyboard on/off
- **Swipe left/right** - Switch between platforms (Google, YouTube, Instagram)
- **Two fingers up** - Toggle high contrast mode
- **Two fingers down** - Toggle voice feedback

### Keyboard Shortcuts
- **Q** - Quit application
- **T** - Start custom gesture training mode
- **S** - Save trained gesture model
- **C** - Toggle high contrast mode
- **V** - Toggle voice feedback
- **L** - Switch keyboard language

## 🌐 Chrome Extension Setup

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked"
4. Select the repository directory
5. The files are named `chrome-extensionmanifest.json`, `chrome-extensionpopup.html`, and `chrome-extensioncontent.js`

**Note:** You may need to rename these files by removing the "chrome-extension" prefix:
```bash
mv chrome-extensionmanifest.json manifest.json
mv chrome-extensionpopup.html popup.html
mv chrome-extensioncontent.js content.js
```

Then load the directory as an unpacked extension.

## 🎨 Features

- **Multi-language Support** - English, Spanish, French, German keyboards
- **Platform Integration** - Search on Google, YouTube, Instagram
- **Accessibility** - High contrast mode and voice feedback
- **Custom Gestures** - Train your own gesture commands
- **Mobile Ready** - Kivy app for Android devices

## 📱 Mobile App Features

The mobile application provides:
- Real-time camera preview
- Gesture detection toggle
- Swipe gesture recognition
- Touch-friendly interface

## 🔧 Troubleshooting

### Camera Not Working
- Check camera permissions
- Ensure no other application is using the camera
- Try a different camera index if you have multiple cameras

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Try upgrading pip: `pip install --upgrade pip`

### Chrome Extension Issues
- Ensure the desktop app is running for WebSocket connection
- Check that files are properly renamed (remove "chrome-extension" prefix)
- Verify Developer mode is enabled in Chrome

### Performance Issues
- Close other applications using the camera
- Reduce video quality settings if needed
- Ensure good lighting for better hand detection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Special thanks to **Muhammad Usama** for mentorship and guidance throughout this project.

---

For more information, see the main [README.md](README.md)
