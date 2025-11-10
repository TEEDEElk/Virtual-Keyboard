#!/usr/bin/env python3
"""
Simple icon generator for Chrome Extension
Creates basic placeholder icons if PIL/Pillow is available
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    def create_icon(size, filename):
        """Create a simple hand icon"""
        # Create image with gradient background
        img = Image.new('RGBA', (size, size), (103, 126, 234, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw a simple hand emoji or shape
        # Simple hand outline
        hand_color = (255, 255, 255, 255)
        
        if size >= 128:
            # Draw a detailed hand for large icon
            # Palm
            draw.ellipse([size*0.2, size*0.35, size*0.8, size*0.85], fill=hand_color)
            # Fingers
            finger_width = size * 0.12
            finger_positions = [0.25, 0.38, 0.5, 0.62, 0.72]
            for pos in finger_positions:
                draw.ellipse([
                    size*pos - finger_width/2, size*0.15,
                    size*pos + finger_width/2, size*0.45
                ], fill=hand_color)
        elif size >= 48:
            # Medium detail for 48px
            draw.ellipse([size*0.25, size*0.4, size*0.75, size*0.85], fill=hand_color)
            # Three fingers
            for pos in [0.35, 0.5, 0.65]:
                draw.ellipse([
                    size*pos - size*0.08, size*0.2,
                    size*pos + size*0.08, size*0.5
                ], fill=hand_color)
        else:
            # Simple shape for 16px
            draw.ellipse([size*0.3, size*0.4, size*0.7, size*0.8], fill=hand_color)
            draw.rectangle([size*0.4, size*0.2, size*0.6, size*0.5], fill=hand_color)
        
        # Save icon
        img.save(filename, 'PNG')
        print(f"✅ Created {filename}")
    
    def main():
        print("🎨 Creating Chrome Extension Icons...")
        print()
        
        # Create icons directory
        icons_dir = "chrome-extension/icons"
        os.makedirs(icons_dir, exist_ok=True)
        
        # Create icons
        sizes = [(16, "icon16.png"), (48, "icon48.png"), (128, "icon128.png")]
        
        for size, filename in sizes:
            filepath = os.path.join(icons_dir, filename)
            create_icon(size, filepath)
        
        print()
        print("✨ All icons created successfully!")
        print(f"📁 Icons saved to: {icons_dir}/")
        print()
        print("Note: These are simple placeholder icons.")
        print("For a professional look, consider creating custom icons")
        print("or using an online icon generator.")
    
    if __name__ == "__main__":
        main()

except ImportError:
    print("❌ PIL/Pillow not installed.")
    print()
    print("To create icons automatically, install Pillow:")
    print("  pip install Pillow")
    print()
    print("Or create icons manually:")
    print("  1. Create 3 PNG images: 16x16, 48x48, 128x128 pixels")
    print("  2. Name them: icon16.png, icon48.png, icon128.png")
    print("  3. Save to: chrome-extension/icons/")
    print()
    print("You can use any image editor or online icon generator.")
