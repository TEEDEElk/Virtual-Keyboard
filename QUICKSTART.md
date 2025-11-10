# 🚀 Quick Start Guide

Get your gesture-controlled virtual keyboard running in 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

This will install all required Python packages including OpenCV, MediaPipe, TensorFlow, and more.

## Step 2: Test the Application (1 minute)

```bash
python main.py
```

### What to Expect:
- A window will open showing your webcam feed
- Make gestures with your hand in front of the camera
- Watch for hand tracking landmarks (colored lines on your hand)

### First Gestures to Try:
1. **Open your hand** (all fingers extended) and hold for 3 seconds → This toggles the keyboard
2. **Pinch** (bring thumb and index finger together) → This selects keys when keyboard is visible
3. **Swipe** (move your hand left or right with fingers extended) → This switches platforms

## Step 3: Use the Virtual Keyboard (2 minutes)

Once the application is running:

1. **Toggle Keyboard**: Hold your hand open (palm facing camera) for 3 seconds
2. **Type**: Pinch to select keys (bring thumb and index finger together)
3. **Search**: Type your query and it will auto-search after 3 seconds of no input
4. **Switch Platform**: Press P key or swipe gesture to cycle through Google/YouTube/Instagram

## Platform Switching

The keyboard can search on different platforms:
- **Google** 🔍 - Web search
- **YouTube** 📺 - Video search  
- **Instagram** 📷 - Hashtag search
- **Browser** 🌐 - Bing search

## Keyboard Shortcuts

While the application is running:
- `Q` - Quit
- `T` - Train custom gestures
- `C` - Toggle high contrast mode
- `V` - Toggle voice feedback
- `L` - Switch keyboard language

## Troubleshooting

### Camera not detected?
```python
# Edit main.py line 34 to try different camera
self.cap = cv2.VideoCapture(1)  # Try 1, 2, 3... instead of 0
```

### Hand not detected?
- Ensure good lighting
- Keep your hand at arm's length from camera
- Make sure hand is clearly visible (no obstructions)

### Module import errors?
```bash
# Install specific packages that are missing
pip install opencv-python mediapipe pyautogui
```

## Chrome Extension (Optional)

To use the browser extension:

### Linux/Mac:
```bash
./setup_extension.sh
```

### Windows:
```bash
setup_extension.bat
```

Then follow the on-screen instructions to load it in Chrome.

## Mobile App (Optional)

For Android users:

```bash
# Run on desktop first to test
python mobile_app.py

# Or build APK
buildozer android debug
```

## Need Help?

- Check [SETUP.md](SETUP.md) for detailed installation guide
- Review [README.md](README.md) for project overview
- Check the issues page on GitHub

---

**Enjoy your gesture-controlled keyboard! 🖐️⌨️**
