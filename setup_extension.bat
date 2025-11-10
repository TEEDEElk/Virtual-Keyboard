@echo off
REM Setup script for Chrome Extension (Windows)

echo Setting up Chrome Extension...
echo.

REM Create chrome-extension directory if it doesn't exist
if not exist "chrome-extension" (
    mkdir chrome-extension
    echo Created chrome-extension directory
)

REM Copy and rename files
echo Copying extension files...
copy chrome-extensionmanifest.json chrome-extension\manifest.json
copy chrome-extensionpopup.html chrome-extension\popup.html
copy chrome-extensioncontent.js chrome-extension\content.js

echo Files copied successfully
echo.

REM Create icons directory with placeholder info
mkdir chrome-extension\icons 2>nul

echo Icon Requirements for Chrome Extension > chrome-extension\icons\README.txt
echo ======================================= >> chrome-extension\icons\README.txt
echo. >> chrome-extension\icons\README.txt
echo You need to add three icon files here: >> chrome-extension\icons\README.txt
echo - icon16.png (16x16 pixels) >> chrome-extension\icons\README.txt
echo - icon48.png (48x48 pixels) >> chrome-extension\icons\README.txt
echo - icon128.png (128x128 pixels) >> chrome-extension\icons\README.txt
echo. >> chrome-extension\icons\README.txt
echo You can: >> chrome-extension\icons\README.txt
echo 1. Create your own icons with an image editor >> chrome-extension\icons\README.txt
echo 2. Use an icon generator online >> chrome-extension\icons\README.txt
echo 3. Use placeholder icons temporarily >> chrome-extension\icons\README.txt

echo Created icons directory with instructions
echo.

echo Setup Complete!
echo.
echo Next steps:
echo 1. Add icon images to chrome-extension\icons\
echo 2. Open Chrome and go to chrome://extensions/
echo 3. Enable 'Developer mode'
echo 4. Click 'Load unpacked' and select the 'chrome-extension' folder
echo.
echo Happy gesture controlling!

pause
