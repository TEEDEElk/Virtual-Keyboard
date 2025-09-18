# Gesture-Controlled Virtual Keyboard

The Gesture-Controlled Virtual Keyboard is a real-time system that replaces traditional input with hand gestures. Using Python, OpenCV, and MediaPipe, it tracks fingertip positions to detect tap gestures on a dynamic on-screen keyboard. Users can type and perform searches on Google, YouTube, and Instagram, with queries executed automatically. This touchless, intuitive solution has applications in accessibility, public kiosks, education, and creative interactive systems.

## Features

### Core Features
- **Real-time hand gesture recognition** using MediaPipe
- **Dynamic virtual keyboard** with multi-language support
- **Automatic search execution** on multiple platforms (Google, YouTube, Instagram)
- **Touchless interaction** for improved accessibility
- **Voice feedback** for enhanced user experience
- **High contrast mode** for visual accessibility
- **Custom gesture training** and recognition

### Gesture Controls
- **Pinch (Thumb + Index)**: Type/Select keys on virtual keyboard
- **Swipe left/right**: Switch between platforms (Google/YouTube/Instagram)
- **Hand open for 3 seconds**: Toggle keyboard visibility
- **Two fingers up**: Toggle high contrast mode
- **Two fingers down**: Toggle voice feedback

### Language Support
- English (QWERTY)
- Spanish (QWERTY + ñ)
- French (AZERTY)
- German (QWERTZ)

### Platform Integration
- **Google Search**: Direct web search queries
- **YouTube**: Video search functionality
- **Instagram**: Hashtag and content exploration
- **Browser**: General web search (Bing)

## Installation

### Prerequisites
- Python 3.7 or higher
- Webcam/Camera
- Internet connection (for search functionality and TTS)

### Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python>=4.5.0
- mediapipe>=0.8.0
- pyautogui>=0.9.52
- numpy>=1.21.0
- tensorflow>=2.7.0
- gtts>=2.2.0
- pygame>=2.0.0
- scikit-learn>=1.0.0
- kivy>=2.1.0 (for mobile app)

## Usage

### Desktop Application
```bash
python main.py
```

### Mobile Application
```bash
python mobile_app.py
```

## Keyboard Shortcuts

While the application is running, you can use these keyboard shortcuts:

- **'q'**: Quit application
- **'t'**: Start gesture training mode
- **'s'**: Save trained gesture model (while in training mode)
- **'c'**: Toggle high contrast mode
- **'v'**: Toggle voice feedback
- **'l'**: Switch language

## Gesture Training

The application supports custom gesture training:

1. Press 't' to enter training mode
2. Enter a gesture name when prompted
3. Perform the gesture multiple times (10+ samples recommended)
4. Press 's' to save the trained model
5. The gesture will be recognized in future sessions

## Configuration

### Keyboard Layouts
Keyboard layouts are stored in `keyboard_layouts/` directory as JSON files:
- `en.json` - English QWERTY
- `es.json` - Spanish QWERTY
- `fr.json` - French AZERTY
- `de.json` - German QWERTZ

### Customization
You can modify the following parameters in `main.py`:
- `input_timeout`: Auto-search delay (default: 3 seconds)
- `min_detection_confidence`: Hand detection sensitivity (default: 0.7)
- `min_tracking_confidence`: Hand tracking stability (default: 0.7)
- Platform colors and URLs

## Accessibility Features

### Visual Accessibility
- **High contrast mode**: Black background with white keys for better visibility
- **Platform color coding**: Different colors for each platform (Google blue, YouTube red, etc.)
- **Real-time visual feedback**: Key highlighting and gesture indicators

### Audio Accessibility
- **Text-to-speech feedback**: Announces typed characters and actions
- **Multilingual TTS**: Supports multiple languages for voice feedback
- **Audio confirmations**: Platform switches and mode changes

### Motor Accessibility
- **Touchless operation**: Complete hands-free interaction
- **Adjustable sensitivity**: Configurable gesture detection thresholds
- **Alternative input methods**: Keyboard shortcuts for all functions

## Architecture

### Core Components
1. **GestureKeyboard**: Main application class
2. **Hand Detection**: MediaPipe-based hand landmark detection
3. **Gesture Recognition**: Custom and pre-defined gesture classification
4. **Virtual Keyboard**: Dynamic on-screen keyboard rendering
5. **Platform Integration**: Web search and navigation
6. **Voice System**: Text-to-speech feedback

### File Structure
```
Virtual-Keyboard/
├── main.py                 # Main desktop application
├── mobile_app.py          # Mobile/tablet application
├── requirements.txt       # Python dependencies
├── keyboard_layouts/      # Language-specific layouts
│   ├── en.json           # English layout
│   ├── es.json           # Spanish layout
│   ├── fr.json           # French layout
│   └── de.json           # German layout
├── models/               # Trained gesture models (created at runtime)
│   ├── custom_gestures.h5
│   └── custom_gestures.json
└── README.md             # This file
```

## Performance Optimization

### Real-time Processing
- **Frame rate optimization**: Efficient OpenCV processing
- **Gesture caching**: Reduces computation overhead
- **Threaded auto-search**: Non-blocking search execution

### Memory Management
- **Model loading**: Lazy loading of TensorFlow models
- **Resource cleanup**: Proper camera and window disposal
- **Garbage collection**: Efficient memory usage patterns

## Use Cases

### Accessibility
- **Motor disabilities**: Hands-free computer interaction
- **Public spaces**: Touchless interfaces in kiosks
- **Medical environments**: Sterile interaction systems
- **Assistive technology**: Integration with existing accessibility tools

### Education
- **Interactive learning**: Gesture-based educational tools
- **Language learning**: Multi-language keyboard practice
- **STEM education**: Computer vision and AI demonstrations
- **Special needs education**: Alternative input methods

### Creative Applications
- **Art installations**: Interactive gesture-controlled displays
- **Gaming**: Motion-controlled game interfaces
- **Presentation tools**: Hands-free slide navigation
- **Smart home**: Gesture-controlled device interaction

## Troubleshooting

### Common Issues

**Camera not detected:**
- Ensure webcam is connected and not used by other applications
- Check camera permissions in system settings

**Poor gesture recognition:**
- Ensure good lighting conditions
- Keep hand within camera frame
- Calibrate gesture sensitivity in code

**TTS not working:**
- Check internet connection for gTTS
- Verify audio output device
- Install required audio codecs

**Keyboard not visible:**
- Use hand open gesture for 3 seconds
- Press spacebar to toggle visibility
- Check application window focus

### Performance Issues
- Close other camera applications
- Reduce video resolution if needed
- Adjust detection confidence parameters
- Use dedicated GPU if available

## Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install development dependencies
4. Run tests and linting
5. Submit pull requests

### Adding New Languages
1. Create new layout file in `keyboard_layouts/`
2. Add language enum in `main.py`
3. Test with various input scenarios
4. Update documentation

### Extending Gestures
1. Use training mode to collect gesture data
2. Implement gesture recognition logic
3. Add visual and audio feedback
4. Document new gesture controls

## License

This project is open source and available under the MIT License.

## Future Enhancements

- **Eye tracking integration**: Gaze-based cursor control
- **Voice commands**: Speech recognition for advanced controls
- **Cloud synchronization**: Cross-device gesture model sharing
- **AR/VR support**: Extended reality interface integration
- **Mobile gestures**: Smartphone camera gesture recognition
- **Multi-hand support**: Two-hand gesture combinations
- **Gesture macros**: Complex gesture sequences
- **Adaptive learning**: User-specific gesture optimization

## Support

For issues, feature requests, or contributions, please visit the project repository or contact the development team.