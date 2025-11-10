# 📁 Project Structure

```
Virtual-Keyboard/
│
├── 📄 main.py                          # Main desktop application
├── 📄 mobile_app.py                    # Kivy mobile application  
├── 📄 my_script.py                     # Project structure outline
├── 📄 create_icons.py                  # Icon generator utility
│
├── 📄 requirements.txt                 # Python dependencies
├── 📄 buildozer.spec                   # Android build configuration
│
├── 📁 keyboard_layouts/                # Multi-language keyboard layouts
│   ├── en.json                        # English QWERTY
│   ├── es.json                        # Spanish with ñ
│   ├── fr.json                        # French AZERTY  
│   └── de.json                        # German QWERTZ
│
├── 📁 models/                          # ML gesture models
│   ├── README.md                      # Model training guide
│   ├── custom_gestures.h5             # Trained model (generated)
│   └── custom_gestures.json           # Gesture classes (generated)
│
├── 📁 chrome-extension/                # Browser extension (generated)
│   ├── manifest.json                  # Extension config
│   ├── popup.html                     # Extension UI
│   ├── content.js                     # Content script
│   └── icons/                         # Extension icons
│       ├── icon16.png                 # 16x16 icon
│       ├── icon48.png                 # 48x48 icon
│       └── icon128.png                # 128x128 icon
│
├── 📄 chrome-extensionmanifest.json    # Extension manifest (source)
├── 📄 chrome-extensionpopup.html       # Extension popup (source)
├── 📄 chrome-extensioncontent.js       # Extension script (source)
│
├── 📄 setup_extension.sh               # Linux/Mac extension setup
├── 📄 setup_extension.bat              # Windows extension setup
│
├── 📄 README.md                        # Project overview
├── 📄 SETUP.md                         # Detailed setup guide
├── 📄 QUICKSTART.md                    # Quick start guide
├── 📄 STRUCTURE.md                     # This file
│
└── 📄 .gitignore                       # Git ignore rules
```

## 📚 File Descriptions

### Core Application Files

#### `main.py`
The main desktop application featuring:
- Real-time hand tracking with MediaPipe
- Virtual on-screen keyboard
- Multi-platform search (Google, YouTube, Instagram)
- Custom gesture training
- Voice feedback with gTTS
- High contrast mode
- Multi-language support

**Key Classes:**
- `Platform(Enum)` - Search platform types
- `Language(Enum)` - Supported languages
- `GestureKeyboard` - Main application class

#### `mobile_app.py`
Kivy-based mobile application for Android:
- Camera integration
- Hand tracking
- Gesture recognition
- Touch-friendly interface

#### `create_icons.py`
Utility to generate placeholder icons for Chrome extension:
- Creates 16x16, 48x48, 128x128 PNG icons
- Requires PIL/Pillow
- Simple hand icon design

### Configuration Files

#### `requirements.txt`
Python package dependencies:
- opencv-python - Computer vision
- mediapipe - Hand tracking
- pyautogui - System control
- tensorflow - ML models
- gtts - Text-to-speech
- pygame - Audio playback
- scikit-learn - ML utilities
- kivy - Mobile GUI

#### `buildozer.spec`
Android build configuration for Kivy application

#### `.gitignore`
Git ignore rules:
- Python cache files
- Virtual environments
- Trained models (large files)
- Build artifacts
- IDE files

### Data Files

#### `keyboard_layouts/*.json`
Keyboard layouts for different languages in JSON format.
Each file contains a 2D array representing keyboard rows and keys.

**Format:**
```json
[
  ["1", "2", "3", ...],
  ["q", "w", "e", ...],
  ...
]
```

#### `models/custom_gestures.h5`
Trained TensorFlow/Keras model (generated after training)

#### `models/custom_gestures.json`
List of gesture class names (generated after training)

### Chrome Extension Files

#### `chrome-extensionmanifest.json` (source)
Chrome extension manifest v3 configuration:
- Extension metadata
- Permissions
- Content script definitions
- Icon references

#### `chrome-extensionpopup.html` (source)
Extension popup UI:
- Connection status indicator
- Platform selector
- Settings toggles
- Gesture reference guide

#### `chrome-extensioncontent.js` (source)
Content script that runs on web pages:
- WebSocket connection to desktop app
- Gesture data handling
- Text input integration
- Search URL generation

#### `chrome-extension/` (generated)
Built extension directory created by setup scripts.
Load this directory in Chrome as an unpacked extension.

### Setup Scripts

#### `setup_extension.sh` (Linux/Mac)
Bash script that:
1. Creates chrome-extension directory
2. Copies and renames extension files
3. Creates icons directory with instructions

#### `setup_extension.bat` (Windows)
Batch script with same functionality for Windows

### Documentation Files

#### `README.md`
Project overview with features and use cases

#### `SETUP.md`
Comprehensive installation and setup guide:
- Prerequisites
- Installation steps
- Chrome extension setup
- Troubleshooting

#### `QUICKSTART.md`
5-minute quick start guide for getting up and running

#### `STRUCTURE.md`
This file - project structure documentation

## 🔄 Workflow

### Development Workflow
1. Edit source files (main.py, chrome-extension*.*)
2. Test changes
3. Run setup script to update extension
4. Reload extension in Chrome

### Extension Setup Workflow
1. Run `setup_extension.sh` or `setup_extension.bat`
2. (Optional) Run `create_icons.py` to generate icons
3. Load `chrome-extension/` in Chrome

### Custom Gesture Training Workflow
1. Run `python main.py`
2. Press 'T' to enter training mode
3. Perform gesture multiple times
4. Press 'S' to train and save model
5. Models saved to `models/` directory

## 🎯 Entry Points

- **Desktop App**: `python main.py`
- **Mobile App**: `python mobile_app.py`
- **Extension Setup**: `./setup_extension.sh` or `setup_extension.bat`
- **Icon Generator**: `python create_icons.py`

## 📦 Generated Files

These files are created during runtime or setup:

- `models/custom_gestures.h5` - After gesture training
- `models/custom_gestures.json` - After gesture training
- `chrome-extension/` - After running setup script
- `__pycache__/` - Python bytecode cache
- `.buildozer/` - Buildozer build cache

## 🔒 Ignored Files

See `.gitignore` for complete list of files excluded from version control.
