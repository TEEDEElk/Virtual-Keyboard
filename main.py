import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import webbrowser
import time
import threading
import json
import os
import tensorflow as tf
from tensorflow import keras
from enum import Enum
from gtts import gTTS
import pygame
from pygame import mixer
import tempfile
from sklearn.model_selection import train_test_split

class Platform(Enum):
    GOOGLE = 1
    YOUTUBE = 2
    INSTAGRAM = 3
    BROWSER = 4  # For browser control

class Language(Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"

class GestureKeyboard:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Application state
        self.current_platform = Platform.GOOGLE
        self.current_language = Language.ENGLISH
        self.keyboard_visible = False
        self.text_buffer = ""
        self.last_input_time = 0
        self.input_timeout = 3.0  # seconds before auto-search
        self.high_contrast_mode = False
        self.voice_feedback = True
        
        # Load keyboard layout
        self.load_keyboard_layout()
        
        # Platform URLs
        self.platform_urls = {
            Platform.GOOGLE: "https://www.google.com/search?q=",
            Platform.YOUTUBE: "https://www.youtube.com/results?search_query=",
            Platform.INSTAGRAM: "https://www.instagram.com/explore/tags/",
            Platform.BROWSER: "https://www.bing.com/search?q="  # Default browser search
        }
        
        # Platform colors
        self.platform_colors = {
            Platform.GOOGLE: (66, 133, 244),      # Google blue
            Platform.YOUTUBE: (255, 0, 0),        # YouTube red
            Platform.INSTAGRAM: (193, 53, 132),   # Instagram purple
            Platform.BROWSER: (0, 200, 83)        # Browser green
        }
        
        # Initialize text-to-speech
        mixer.init()
        
        # Load custom gesture model if available
        self.custom_gesture_model = None
        self.custom_gesture_classes = []
        self.load_custom_gesture_model()
        
        # Thread for auto-search
        self.auto_search_thread = None
        self.running = True
        
        # Custom gesture training
        self.training_mode = False
        self.current_gesture_class = None
        self.training_data = []
        self.training_labels = []
        
    def load_keyboard_layout(self):
        """Load keyboard layout for current language"""
        layout_file = f"keyboard_layouts/{self.current_language.value}.json"
        try:
            with open(layout_file, 'r') as f:
                self.keyboard_rows = json.load(f)
        except FileNotFoundError:
            # Default English layout
            self.keyboard_rows = [
                ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
                ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
                ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
                ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.'],
                ['Space', 'Back', 'Clear', 'Search', 'Lang']
            ]
        
        # Keyboard dimensions
        self.keyboard_width = min(800, self.screen_width - 40)
        self.key_width = 70
        self.key_height = 50
        self.key_margin = 10
        self.keyboard_height = len(self.keyboard_rows) * (self.key_height + self.key_margin) + 40
        
    def load_custom_gesture_model(self):
        """Load trained custom gesture model"""
        try:
            self.custom_gesture_model = keras.models.load_model('models/custom_gestures.h5')
            with open('models/custom_gestures.json', 'r') as f:
                self.custom_gesture_classes = json.load(f)
            print(f"Loaded custom gesture model with {len(self.custom_gesture_classes)} classes")
        except (OSError, IOError):
            print("No custom gesture model found. Using basic gestures.")
    
    def speak_text(self, text):
        """Convert text to speech (if voice feedback enabled)"""
        if not self.voice_feedback:
            return
            
        try:
            tts = gTTS(text=text, lang=self.current_language.value)
            with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                mixer.music.load(tmp_file.name)
                mixer.music.play()
                # Wait for playback to finish
                while mixer.music.get_busy():
                    time.sleep(0.1)
        except Exception as e:
            print(f"Text-to-speech error: {e}")
    
    def draw_keyboard(self, image):
        """Draw the virtual keyboard on the image"""
        if not self.keyboard_visible:
            return
            
        # Calculate keyboard position (centered at bottom)
        keyboard_x = (image.shape[1] - self.keyboard_width) // 2
        keyboard_y = image.shape[0] - self.keyboard_height - 20
        
        # Choose colors based on contrast mode
        if self.high_contrast_mode:
            bg_color = (0, 0, 0)  # Black background
            key_color = (255, 255, 255)  # White keys
            text_color = (0, 0, 0)  # Black text
            border_color = (255, 255, 255)  # White border
        else:
            bg_color = (50, 50, 50)  # Dark gray background
            key_color = (200, 200, 200)  # Light gray keys
            text_color = (0, 0, 0)  # Black text
            border_color = (200, 200, 200)  # Light gray border
        
        # Draw keyboard background
        cv2.rectangle(image, 
                     (keyboard_x, keyboard_y), 
                     (keyboard_x + self.keyboard_width, keyboard_y + self.keyboard_height),
                     bg_color, -1)
        cv2.rectangle(image, 
                     (keyboard_x, keyboard_y), 
                     (keyboard_x + self.keyboard_width, keyboard_y + self.keyboard_height),
                     border_color, 2)
        
        # Draw keys for each row
        for row_idx, row in enumerate(self.keyboard_rows):
            row_y = keyboard_y + 20 + row_idx * (self.key_height + self.key_margin)
            
            # Calculate row width to center it
            row_width = len(row) * (self.key_width + self.key_margin) - self.key_margin
            row_x = keyboard_x + (self.keyboard_width - row_width) // 2
            
            for key_idx, key in enumerate(row):
                key_x = row_x + key_idx * (self.key_width + self.key_margin)
                key_y = row_y
                
                # Draw key with special colors for important buttons
                if key in ['Space', 'Search']:
                    key_fill_color = self.platform_colors[self.current_platform]
                elif key == 'Lang':
                    key_fill_color = (0, 128, 255)  # Blue for language switch
                else:
                    key_fill_color = key_color
                    
                cv2.rectangle(image, 
                             (key_x, key_y), 
                             (key_x + self.key_width, key_y + self.key_height),
                             key_fill_color, -1)
                cv2.rectangle(image, 
                             (key_x, key_y), 
                             (key_x + self.key_width, key_y + self.key_height),
                             border_color, 1)
                
                # Draw key label
                font_scale = 0.7 if len(key) <= 1 else 0.5
                text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                text_x = key_x + (self.key_width - text_size[0]) // 2
                text_y = key_y + (self.key_height + text_size[1]) // 2
                
                cv2.putText(image, key, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
    
    def get_key_at_position(self, x, y):
        """Get the key at the given screen position"""
        if not self.keyboard_visible:
            return None
            
        # Calculate keyboard position
        keyboard_x = (self.screen_width - self.keyboard_width) // 2
        keyboard_y = self.screen_height - self.keyboard_height - 20
        
        # Check if position is within keyboard bounds
        if not (keyboard_x <= x <= keyboard_x + self.keyboard_width and
                keyboard_y <= y <= keyboard_y + self.keyboard_height):
            return None
            
        # Find which key was pressed
        for row_idx, row in enumerate(self.keyboard_rows):
            row_y = keyboard_y + 20 + row_idx * (self.key_height + self.key_margin)
            
            # Calculate row width to center it
            row_width = len(row) * (self.key_width + self.key_margin) - self.key_margin
            row_x = keyboard_x + (self.keyboard_width - row_width) // 2
            
            for key_idx, key in enumerate(row):
                key_x = row_x + key_idx * (self.key_width + self.key_margin)
                key_y = row_y
                
                if (key_x <= x <= key_x + self.key_width and
                    key_y <= y <= key_y + self.key_height):
                    return key
                    
        return None
    
    def handle_key_press(self, key):
        """Handle a key press from the virtual keyboard"""
        if key == 'Space':
            self.text_buffer += ' '
            if self.voice_feedback:
                self.speak_text("space")
        elif key == 'Back':
            self.text_buffer = self.text_buffer[:-1]
            if self.voice_feedback:
                self.speak_text("backspace")
        elif key == 'Clear':
            self.text_buffer = ""
            if self.voice_feedback:
                self.speak_text("clear")
        elif key == 'Search':
            self.perform_search()
        elif key == 'Lang':
            self.switch_language()
        else:
            self.text_buffer += key
            if self.voice_feedback and len(key) == 1:
                self.speak_text(key)
            
        self.last_input_time = time.time()
        
        # Start auto-search timer
        if self.auto_search_thread is None or not self.auto_search_thread.is_alive():
            self.auto_search_thread = threading.Thread(target=self.auto_search_timer)
            self.auto_search_thread.daemon = True
            self.auto_search_thread.start()
    
    def auto_search_timer(self):
        """Timer for auto-search after no input"""
        while self.running and time.time() - self.last_input_time < self.input_timeout:
            time.sleep(0.1)
            
        if time.time() - self.last_input_time >= self.input_timeout and self.text_buffer:
            self.perform_search()
    
    def perform_search(self):
        """Perform the search on the selected platform"""
        if not self.text_buffer:
            return
            
        search_url = self.platform_urls[self.current_platform] + self.text_buffer.replace(' ', '+')
        webbrowser.open(search_url)
        
        if self.voice_feedback:
            self.speak_text(f"Searching {self.current_platform.name} for {self.text_buffer}")
            
        self.text_buffer = ""
        self.keyboard_visible = False
    
    def switch_language(self):
        """Switch to the next language"""
        languages = list(Language)
        current_idx = languages.index(self.current_language)
        self.current_language = languages[(current_idx + 1) % len(languages)]
        self.load_keyboard_layout()
        
        if self.voice_feedback:
            self.speak_text(f"Switched to {self.current_language.name}")
    
    def check_pinch_gesture(self, landmarks):
        """Check if a pinch gesture is detected between thumb and index finger"""
        if not landmarks:
            return False
            
        # Get thumb tip and index finger tip landmarks
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate distance between thumb and index finger
        distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
        
        # Pinch detected if distance is below threshold
        return distance < 0.05
    
    def check_platform_switch_gesture(self, landmarks):
        """Check for platform switch gesture (swipe left/right)"""
        if not landmarks or len(landmarks) < 9:
            return False
            
        # Get wrist and middle finger landmarks for swipe detection
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        
        # Check for horizontal movement (swipe)
        if abs(wrist.x - middle_tip.x) > 0.2:
            return True
            
        return False
    
    def switch_platform(self):
        """Switch to the next platform"""
        platforms = list(Platform)
        current_idx = platforms.index(self.current_platform)
        self.current_platform = platforms[(current_idx + 1) % len(platforms)]
        
        if self.voice_feedback:
            self.speak_text(f"Switched to {self.current_platform.name}")
    
    def recognize_custom_gesture(self, landmarks):
        """Recognize custom gestures using trained model"""
        if not self.custom_gesture_model or not landmarks:
            return None
            
        # Prepare input data
        sample = []
        for lm in landmarks:
            sample.extend([lm.x, lm.y, lm.z])
            
        sample = np.array([sample])
        
        # Predict gesture
        prediction = self.custom_gesture_model.predict(sample, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        
        # Only return if confidence is high enough
        if confidence > 0.7 and predicted_class_idx < len(self.custom_gesture_classes):
            return self.custom_gesture_classes[predicted_class_idx]
        
        return None
    
    def start_gesture_training(self, gesture_class):
        """Start training mode for a new gesture"""
        self.training_mode = True
        self.current_gesture_class = gesture_class
        self.training_data = []
        self.training_labels = []
        
        if self.voice_feedback:
            self.speak_text(f"Training mode started for {gesture_class}. Show the gesture multiple times.")
    
    def add_training_sample(self, landmarks):
        """Add a sample to training data"""
        if not self.training_mode or not landmarks:
            return
            
        # Prepare sample
        sample = []
        for lm in landmarks:
            sample.extend([lm.x, lm.y, lm.z])
            
        self.training_data.append(sample)
        self.training_labels.append(self.current_gesture_class)
        
        if self.voice_feedback and len(self.training_data) % 10 == 0:
            self.speak_text(f"{len(self.training_data)} samples collected")
    
    def train_gesture_model(self):
        """Train the custom gesture model"""
        if not self.training_data or not self.training_labels:
            return False
            
        # Convert to numpy arrays
        X = np.array(self.training_data)
        
        # Create label mapping
        unique_labels = list(set(self.training_labels))
        y = np.array([unique_labels.index(label) for label in self.training_labels])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build model
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(len(unique_labels), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)
        
        # Save model and classes
        os.makedirs('models', exist_ok=True)
        model.save('models/custom_gestures.h5')
        with open('models/custom_gestures.json', 'w') as f:
            json.dump(unique_labels, f)
        
        # Update current model
        self.custom_gesture_model = model
        self.custom_gesture_classes = unique_labels
        
        if self.voice_feedback:
            self.speak_text(f"Model trained with {len(unique_labels)} gestures")
            
        return True
    
    def toggle_high_contrast(self):
        """Toggle high contrast mode"""
        self.high_contrast_mode = not self.high_contrast_mode
        if self.voice_feedback:
            status = "on" if self.high_contrast_mode else "off"
            self.speak_text(f"High contrast mode {status}")
    
    def toggle_voice_feedback(self):
        """Toggle voice feedback"""
        self.voice_feedback = not self.voice_feedback
        if self.voice_feedback:
            self.speak_text("Voice feedback on")
        else:
            self.speak_text("Voice feedback off")
    
    def run(self):
        """Main application loop"""
        print("Gesture-Controlled Virtual Keyboard")
        print("Platforms: Google, YouTube, Instagram")
        print("Gestures:")
        print("- Pinch (thumb + index): Type/Select")
        print("- Swipe left/right: Switch platform")
        print("- Hand open for 3s: Toggle keyboard")
        print("- Two fingers up: Toggle high contrast")
        print("- Two fingers down: Toggle voice feedback")
        
        hand_open_start_time = 0
        hand_open_duration = 3.0  # seconds to show hand open to toggle keyboard
        
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                continue
                
            # Flip and convert image
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process hand landmarks
            results = self.hands.process(rgb_image)
            
            # Check for hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                    
                    landmarks = hand_landmarks.landmark
                    
                    # Check for custom gestures first
                    custom_gesture = self.recognize_custom_gesture(landmarks)
                    if custom_gesture:
                        cv2.putText(image, f"Custom: {custom_gesture}", (10, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Check for pinch gesture (typing/selection)
                    if self.check_pinch_gesture(landmarks):
                        # Convert hand position to screen coordinates
                        index_tip = landmarks[8]
                        screen_x = int(index_tip.x * self.screen_width)
                        screen_y = int(index_tip.y * self.screen_height)
                        
                        # Check if keyboard is visible and get pressed key
                        if self.keyboard_visible:
                            key = self.get_key_at_position(screen_x, screen_y)
                            if key:
                                self.handle_key_press(key)
                                # Visual feedback
                                cv2.circle(image, (int(index_tip.x * image.shape[1]), 
                                                  int(index_tip.y * image.shape[0])), 
                                          20, (0, 255, 0), -1)
                        elif self.training_mode:
                            self.add_training_sample(landmarks)
                    
                    # Check for platform switch gesture
                    if self.check_platform_switch_gesture(landmarks):
                        self.switch_platform()
                        # Visual feedback
                        cv2.putText(image, f"Switched to {self.current_platform.name}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                   self.platform_colors[self.current_platform], 2)
                    
                    # Check for hand open gesture to toggle keyboard
                    thumb_tip = landmarks[4]
                    thumb_ip = landmarks[3]
                    index_tip = landmarks[8]
                    index_pip = landmarks[6]
                    middle_tip = landmarks[12]
                    middle_pip = landmarks[10]
                    ring_tip = landmarks[16]
                    ring_pip = landmarks[14]
                    pinky_tip = landmarks[20]
                    pinky_pip = landmarks[18]
                    
                    # Check if all fingers are extended (hand open)
                    fingers_extended = [
                        thumb_tip.x > thumb_ip.x,  # Thumb
                        index_tip.y < index_pip.y,  # Index
                        middle_tip.y < middle_pip.y,  # Middle
                        ring_tip.y < ring_pip.y,  # Ring
                        pinky_tip.y < pinky_pip.y  # Pinky
                    ]
                    
                    if all(fingers_extended):
                        current_time = time.time()
                        if hand_open_start_time == 0:
                            hand_open_start_time = current_time
                        elif current_time - hand_open_start_time >= hand_open_duration:
                            self.keyboard_visible = not self.keyboard_visible
                            hand_open_start_time = 0
                            if self.voice_feedback:
                                status = "visible" if self.keyboard_visible else "hidden"
                                self.speak_text(f"Keyboard {status}")
                    else:
                        hand_open_start_time = 0
                    
                    # Check for two fingers up (high contrast toggle)
                    index_extended = index_tip.y < index_pip.y
                    middle_extended = middle_tip.y < middle_pip.y
                    other_fingers_closed = not (thumb_tip.x > thumb_ip.x or 
                                              ring_tip.y < ring_pip.y or 
                                              pinky_tip.y < pinky_pip.y)
                    
                    if index_extended and middle_extended and other_fingers_closed:
                        self.toggle_high_contrast()
                        time.sleep(1)  # Prevent rapid toggling
                    
                    # Check for two fingers down (voice feedback toggle)
                    index_bent = index_tip.y > index_pip.y
                    middle_bent = middle_tip.y > middle_pip.y
                    other_fingers_closed = not (thumb_tip.x > thumb_ip.x or 
                                              ring_tip.y < ring_pip.y or 
                                              pinky_tip.y < pinky_pip.y)
                    
                    if index_bent and middle_bent and other_fingers_closed:
                        self.toggle_voice_feedback()
                        time.sleep(1)  # Prevent rapid toggling
            
            # Draw keyboard if visible
            if self.keyboard_visible:
                self.draw_keyboard(image)
            
            # Display current platform and text buffer
            platform_name = self.current_platform.name
            platform_color = self.platform_colors[self.current_platform]
            
            cv2.putText(image, f"Platform: {platform_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, platform_color, 2)
            
            cv2.putText(image, f"Language: {self.current_language.name}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            if self.text_buffer:
                cv2.putText(image, f"Text: {self.text_buffer}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display accessibility status
            contrast_status = "ON" if self.high_contrast_mode else "OFF"
            voice_status = "ON" if self.voice_feedback else "OFF"
            
            cv2.putText(image, f"High Contrast: {contrast_status}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            cv2.putText(image, f"Voice Feedback: {voice_status}", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            
            # Display instructions
            instructions = [
                "Pinch to type/select",
                "Swipe to switch platform",
                "Hand open for 3s to toggle keyboard",
                "Two fingers up: Toggle contrast",
                "Two fingers down: Toggle voice"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(image, instruction, (10, image.shape[0] - 30 - i * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Display image
            cv2.imshow('Gesture-Controlled Virtual Keyboard', image)
            
            # Handle keyboard commands
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):  # Start training mode
                gesture_class = input("Enter gesture name: ")
                self.start_gesture_training(gesture_class)
            elif key == ord('s') and self.training_mode:  # Save training
                self.train_gesture_model()
                self.training_mode = False
            elif key == ord('c'):  # Toggle high contrast
                self.toggle_high_contrast()
            elif key == ord('v'):  # Toggle voice feedback
                self.toggle_voice_feedback()
            elif key == ord('l'):  # Switch language
                self.switch_language()
                
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GestureKeyboard()
    app.run()