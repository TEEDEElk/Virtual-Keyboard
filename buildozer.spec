[app]
title = GestureMobileApp
package.name = gesturemobileapp
package.domain = org.example
target = android
source.dir = .
source.include_exts = py,png,jpg,kv,atlas

# Main .py file
main.py = mobile_app.py

# Requirements (comma-separated)
requirements = python3,kivy,opencv-python,mediapipe,numpy

# Permissions
default_permissions = INTERNET,CAMERA

# Orientation
orientation = portrait

# (Optional) Icon
# icon.filename = %(source.dir)s/icon.png

# Android specific
android.permissions = CAMERA,INTERNET
android.api = 33
android.minapi = 21
android.arch = arm64-v8a,armeabi-v7a

# (Optional) Presplash
# presplash.filename = %(source.dir)s/presplash.png

# (Optional) Fullscreen
fullscreen = 1

# (Optional) Log level
log_level = 2

# (Optional) Add any other settings as needed
