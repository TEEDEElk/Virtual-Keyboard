# Models Directory

This directory stores trained gesture recognition models.

## Files
- `custom_gestures.h5` - Trained TensorFlow/Keras model for custom gestures
- `custom_gestures.json` - List of gesture class names

## Training Your Own Gestures

To train custom gestures:

1. Run the main application: `python main.py`
2. Press 't' to enter training mode
3. Enter a gesture name
4. Perform the gesture multiple times (at least 50 samples recommended)
5. Press 's' to save and train the model

The trained model will be automatically saved to this directory.

## Notes
- Model files are gitignored by default due to their size
- The application works with basic gestures even without custom models
- For best results, train in good lighting conditions with clear hand visibility
