#!/bin/bash
# Setup script for Chrome Extension

echo "🔧 Setting up Chrome Extension..."
echo ""

# Create chrome-extension directory if it doesn't exist
if [ ! -d "chrome-extension" ]; then
    mkdir chrome-extension
    echo "✅ Created chrome-extension directory"
fi

# Copy and rename files
echo "📋 Copying extension files..."
cp chrome-extensionmanifest.json chrome-extension/manifest.json
cp chrome-extensionpopup.html chrome-extension/popup.html
cp chrome-extensioncontent.js chrome-extension/content.js

echo "✅ Files copied successfully"
echo ""

# Create icons directory with placeholder info
mkdir -p chrome-extension/icons
cat > chrome-extension/icons/README.txt << 'EOL'
Icon Requirements for Chrome Extension
=======================================

You need to add three icon files here:
- icon16.png (16x16 pixels)
- icon48.png (48x48 pixels)  
- icon128.png (128x128 pixels)

You can:
1. Create your own icons with an image editor
2. Use an icon generator online
3. Use placeholder icons temporarily

For a hand/gesture theme, consider:
- 🖐️ Open hand icon
- ⌨️ Keyboard icon
- 👆 Pointing finger icon

EOL

echo "📁 Created icons directory with instructions"
echo ""

echo "✨ Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Add icon images to chrome-extension/icons/"
echo "2. Open Chrome and go to chrome://extensions/"
echo "3. Enable 'Developer mode'"
echo "4. Click 'Load unpacked' and select the 'chrome-extension' folder"
echo ""
echo "Happy gesture controlling! 🖐️"
