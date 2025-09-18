#!/usr/bin/env python3
"""
Simple demo script to showcase Virtual Keyboard functionality
without requiring camera or heavy dependencies.
"""

import json
import time
from enum import Enum

class Platform(Enum):
    GOOGLE = 1
    YOUTUBE = 2
    INSTAGRAM = 3
    BROWSER = 4

class Language(Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"

class VirtualKeyboardDemo:
    def __init__(self):
        self.current_platform = Platform.GOOGLE
        self.current_language = Language.ENGLISH
        self.keyboard_visible = True
        self.text_buffer = ""
        self.high_contrast_mode = False
        self.voice_feedback = True
        
        self.platform_urls = {
            Platform.GOOGLE: "https://www.google.com/search?q=",
            Platform.YOUTUBE: "https://www.youtube.com/results?search_query=",
            Platform.INSTAGRAM: "https://www.instagram.com/explore/tags/",
            Platform.BROWSER: "https://www.bing.com/search?q="
        }
        
        self.load_keyboard_layout()
    
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
    
    def display_keyboard(self):
        """Display the virtual keyboard in text format"""
        print(f"\n{'='*60}")
        print(f"  VIRTUAL KEYBOARD - {self.current_language.name}")
        print(f"  Platform: {self.current_platform.name}")
        print(f"  High Contrast: {'ON' if self.high_contrast_mode else 'OFF'}")
        print(f"  Voice Feedback: {'ON' if self.voice_feedback else 'OFF'}")
        print(f"{'='*60}")
        
        for i, row in enumerate(self.keyboard_rows):
            # Add spacing for visual effect
            spacing = "  " if i < 4 else " "
            print(f"  {spacing}", end="")
            
            for key in row:
                if key in ['Space', 'Back', 'Clear', 'Search', 'Lang']:
                    print(f"[{key:^7}]", end=" ")
                else:
                    print(f"[{key:^2}]", end=" ")
            print()
        
        print(f"\n  Text Buffer: '{self.text_buffer}'")
        print(f"{'='*60}")
    
    def handle_key_press(self, key):
        """Handle a key press from the virtual keyboard"""
        if key == 'Space':
            self.text_buffer += ' '
            print("  > Added space")
        elif key == 'Back':
            self.text_buffer = self.text_buffer[:-1]
            print("  > Backspace")
        elif key == 'Clear':
            self.text_buffer = ""
            print("  > Cleared text")
        elif key == 'Search':
            self.perform_search()
        elif key == 'Lang':
            self.switch_language()
        else:
            self.text_buffer += key
            print(f"  > Added '{key}'")
    
    def perform_search(self):
        """Simulate search operation"""
        if not self.text_buffer:
            print("  > No text to search!")
            return
            
        search_url = self.platform_urls[self.current_platform] + self.text_buffer.replace(' ', '+')
        print(f"\n  🔍 SEARCHING {self.current_platform.name}:")
        print(f"  Query: '{self.text_buffer}'")
        print(f"  URL: {search_url}")
        print("  > Search would open in browser")
        
        self.text_buffer = ""
    
    def switch_language(self):
        """Switch to the next language"""
        languages = list(Language)
        current_idx = languages.index(self.current_language)
        self.current_language = languages[(current_idx + 1) % len(languages)]
        self.load_keyboard_layout()
        print(f"  > Switched to {self.current_language.name}")
    
    def switch_platform(self):
        """Switch to the next platform"""
        platforms = list(Platform)
        current_idx = platforms.index(self.current_platform)
        self.current_platform = platforms[(current_idx + 1) % len(platforms)]
        print(f"  > Switched to {self.current_platform.name}")
    
    def toggle_high_contrast(self):
        """Toggle high contrast mode"""
        self.high_contrast_mode = not self.high_contrast_mode
        status = "ON" if self.high_contrast_mode else "OFF"
        print(f"  > High contrast mode {status}")
    
    def toggle_voice_feedback(self):
        """Toggle voice feedback"""
        self.voice_feedback = not self.voice_feedback
        status = "ON" if self.voice_feedback else "OFF"
        print(f"  > Voice feedback {status}")
    
    def run_demo(self):
        """Run the interactive demo"""
        print("🎯 GESTURE-CONTROLLED VIRTUAL KEYBOARD DEMO")
        print("=" * 60)
        print("This demo shows the virtual keyboard functionality")
        print("without requiring camera or gesture recognition.")
        print("\nCommands:")
        print("  - Type a key name (e.g., 'h', 'e', 'l', 'l', 'o')")
        print("  - Use 'Space', 'Back', 'Clear', 'Search', 'Lang'")
        print("  - Type 'platform' to switch platform")
        print("  - Type 'contrast' to toggle high contrast")
        print("  - Type 'voice' to toggle voice feedback")
        print("  - Type 'quit' to exit")
        print("=" * 60)
        
        while True:
            self.display_keyboard()
            
            user_input = input("\nEnter command or key: ").strip().lower()
            
            if user_input == 'quit':
                print("\n👋 Demo ended. Thanks for trying the Virtual Keyboard!")
                break
            elif user_input == 'platform':
                self.switch_platform()
            elif user_input == 'contrast':
                self.toggle_high_contrast()
            elif user_input == 'voice':
                self.toggle_voice_feedback()
            elif user_input in ['space', 'back', 'clear', 'search', 'lang']:
                self.handle_key_press(user_input.title())
            elif len(user_input) == 1 and user_input.isalnum():
                self.handle_key_press(user_input)
            elif user_input.isdigit() and len(user_input) == 1:
                self.handle_key_press(user_input)
            elif user_input in [',', '.']:
                self.handle_key_press(user_input)
            else:
                print(f"  ❌ Unknown command: '{user_input}'")
                print("     Try a single character, number, or special command")
            
            time.sleep(0.5)  # Brief pause for readability

if __name__ == "__main__":
    demo = VirtualKeyboardDemo()
    try:
        demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Thanks for trying the Virtual Keyboard!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("Please ensure the keyboard_layouts directory exists with layout files.")