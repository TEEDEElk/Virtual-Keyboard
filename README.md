# ✋ Gesture-Controlled Virtual Keyboard  

> ✅ **Status**: Fully Complete and Ready to Use!

A real-time **gesture-based virtual keyboard** that allows users to perform search queries on **Google, YouTube, and Instagram** using only their hands—no physical input devices required.  

This project uses **computer vision** and **hand tracking** to detect fingertip positions and recognize tap gestures, enabling text input through an on-screen keyboard. Once a query is typed, the system automatically launches a search in the browser.

## 🎉 What's Included

- **Desktop Application** – Full-featured gesture control with multi-language support
- **Chrome Extension** – Browser integration with WebSocket communication
- **Mobile App** – Kivy-based Android application
- **Setup Utilities** – Automated setup scripts and icon generator
- **Comprehensive Documentation** – Quick start, setup guides, and structure docs

👉 **New to this project?** Start with [QUICKSTART.md](QUICKSTART.md) for a 5-minute setup guide!  

---

## 🚀 Features  
- **Real-Time Gesture Input** – Detects fingertip taps using MediaPipe hand landmarks.  
- **Virtual Keyboard** – Dynamic, color-coded keyboard fully rendered with OpenCV.  
- **Multi-Platform Support** – Perform searches on Google, YouTube, Instagram, and Bing.  
- **Multi-Language Keyboards** – Support for English, Spanish, French, and German layouts.
- **Auto Search Execution** – Automatically launches search after a short pause.  
- **Visual Feedback** – Displays typed text, gesture detection, and active platform.
- **Chrome Extension** – Browser integration with WebSocket support.
- **Custom Gesture Training** – Train your own gestures with TensorFlow.
- **Accessibility Features** – High contrast mode and voice feedback.
- **Mobile Support** – Kivy app for Android devices.  

---

## 🛠️ Tech Stack  
- **Python** – Core language for logic and flow.  
- **OpenCV** – Real-time video capture, rendering, and image processing.  
- **MediaPipe** – Hand tracking and landmark detection.  
- **TensorFlow** – Custom gesture recognition with neural networks.
- **NumPy** – Numerical operations (e.g., distance calculations).  
- **gTTS & Pygame** – Text-to-speech and audio feedback.
- **Kivy** – Cross-platform mobile application framework.
- **JavaScript** – Chrome extension with WebSocket communication.

## 📦 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py

# Setup Chrome extension (optional)
./setup_extension.sh  # or setup_extension.bat on Windows
```

For detailed instructions, see [SETUP.md](SETUP.md) or [QUICKSTART.md](QUICKSTART.md).  

---

## 🎯 Real-World Use Cases  
- **Assistive Technology** – Helps users with mobility impairments interact without keyboards or mice.  
- **Public Interfaces** – Safer touchless input for kiosks and smart displays.  
- **Creative Applications** – Interactive installations, education tools, and futuristic UX designs.  

---

## 🔮 Future Improvements  
- ~~Gesture-based scrolling & navigation for search results.~~
- ~~Custom gesture training with ML models.~~ ✅ **Implemented!**
- ~~Multilingual keyboard support.~~ ✅ **Implemented!**
- ~~Chrome extension integration for direct browser control.~~ ✅ **Implemented!**
- Additional language support (Chinese, Japanese, Arabic, etc.)
- Gesture-based scrolling and navigation
- Voice command integration
- Multi-monitor support

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** – Get started in 5 minutes
- **[SETUP.md](SETUP.md)** – Comprehensive installation and setup guide
- **[STRUCTURE.md](STRUCTURE.md)** – Project structure and file organization
- **[COMPLETION.md](COMPLETION.md)** – Project completion checklist and verification  

---

## 🙏 Acknowledgments  
Special thanks to **Muhammad Usama** for mentorship and guidance throughout this project.  

---
