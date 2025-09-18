gesture-keyboard/
├── main.py                      # Main application
├── gesture_trainer.py           # ML model training
├── keyboard_layouts/            # Multilingual keyboard layouts
│   ├── en.json                 # English layout
│   ├── es.json                 # Spanish layout
│   └── fr.json                 # French layout
├── models/                      # Trained gesture models
│   ├── custom_gestures.h5      # Custom gesture model
│   └── custom_gestures.json    # Gesture classes
├── static/                      # Web resources
│   ├── popup.html              # Chrome extension popup
│   ├── content.js              # Content script
│   └── background.js           # Background script
├── chrome-extension/           # Browser extension files
│   ├── manifest.json           # Extension manifest
│   └── icons/                  # Extension icons
└── requirements.txt            # Python dependencies