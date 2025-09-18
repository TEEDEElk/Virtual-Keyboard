from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
import cv2
import numpy as np
import mediapipe as mp

class GestureMobileApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        
        # Camera preview
        self.camera = Camera(resolution=(640, 480), play=True)
        layout.add_widget(self.camera)
        
        # Status label
        self.status_label = Label(text="Gesture Control: Off", size_hint=(1, 0.1))
        layout.add_widget(self.status_label)
        
        # Toggle button
        self.toggle_btn = ToggleButton(text="Start Gesture Control", on_press=self.toggle_control)
        layout.add_widget(self.toggle_btn)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.control_active = False
        
        # Bind camera texture to update method
        self.camera.bind(on_texture=self.update)
        
        return layout
        
    def toggle_control(self, instance):
        self.control_active = instance.state == 'down'
        status = "On" if self.control_active else "Off"
        self.status_label.text = f"Gesture Control: {status}"
        
    def update(self, *args):
        if not self.control_active:
            return
            
        # Get camera texture
        texture = self.camera.texture
        if not texture:
            return
            
        # Convert to OpenCV image
        buffer = texture.pixels
        image = np.frombuffer(buffer, dtype=np.uint8)
        image = image.reshape(texture.height, texture.width, 4)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        # Process image with MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        # Handle gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Simple gesture recognition for mobile
                landmarks = hand_landmarks.landmark
                
                # Check for swipe gestures
                wrist = landmarks[0]
                middle_tip = landmarks[12]
                
                if abs(wrist.x - middle_tip.x) > 0.2:
                    # Horizontal swipe detected
                    if wrist.x < middle_tip.x:
                        self.status_label.text = "Swipe Right →"
                    else:
                        self.status_label.text = "Swipe Left ←"
        
        # Update texture
        texture.blit_buffer(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGBA).tobytes(),
            colorfmt='rgba',
            bufferfmt='ubyte'
        )

if __name__ == '__main__':
    GestureMobileApp().run()