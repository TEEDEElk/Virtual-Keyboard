// Gesture-Controlled Virtual Keyboard Content Script
// This script enables gesture control in the browser

class GestureKeyboardExtension {
    constructor() {
        this.connected = false;
        this.platform = 'google';
        this.voiceFeedback = false;
        this.highContrast = false;
        this.websocket = null;
        this.overlay = null;
        this.keyboardVisible = false;
        
        // Load settings from storage
        this.loadSettings();
        
        // Listen for messages from popup
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            this.handleMessage(request, sender, sendResponse);
            return true; // Keep channel open for async response
        });
    }
    
    loadSettings() {
        chrome.storage.local.get(['platform', 'voiceFeedback', 'highContrast'], (result) => {
            this.platform = result.platform || 'google';
            this.voiceFeedback = result.voiceFeedback || false;
            this.highContrast = result.highContrast || false;
        });
    }
    
    handleMessage(request, sender, sendResponse) {
        switch (request.type) {
            case 'connect':
                this.connect();
                sendResponse({ success: true });
                break;
                
            case 'disconnect':
                this.disconnect();
                sendResponse({ success: true });
                break;
                
            case 'checkStatus':
                sendResponse({ connected: this.connected });
                break;
                
            case 'changePlatform':
                this.platform = request.platform;
                sendResponse({ success: true });
                break;
                
            case 'toggleVoice':
                this.voiceFeedback = request.enabled;
                sendResponse({ success: true });
                break;
                
            case 'toggleContrast':
                this.highContrast = request.enabled;
                if (this.overlay) {
                    this.updateOverlayStyle();
                }
                sendResponse({ success: true });
                break;
                
            default:
                sendResponse({ error: 'Unknown message type' });
        }
    }
    
    connect() {
        if (this.connected) {
            return;
        }
        
        // Try to connect to local Python server via WebSocket
        // The Python application should run a WebSocket server on localhost
        try {
            this.websocket = new WebSocket('ws://localhost:8765');
            
            this.websocket.onopen = () => {
                this.connected = true;
                this.showNotification('Connected to Gesture Keyboard', 'success');
                this.createOverlay();
            };
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleGestureData(data);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showNotification('Connection failed. Make sure the desktop app is running.', 'error');
            };
            
            this.websocket.onclose = () => {
                this.connected = false;
                this.removeOverlay();
                this.showNotification('Disconnected from Gesture Keyboard', 'info');
            };
        } catch (error) {
            console.error('Failed to connect:', error);
            this.showNotification('Connection failed. Make sure the desktop app is running on port 8765.', 'error');
        }
    }
    
    disconnect() {
        if (this.websocket) {
            this.websocket.close();
        }
        this.connected = false;
        this.removeOverlay();
    }
    
    createOverlay() {
        if (this.overlay) {
            return;
        }
        
        // Create overlay container
        this.overlay = document.createElement('div');
        this.overlay.id = 'gesture-keyboard-overlay';
        this.updateOverlayStyle();
        
        // Create status indicator
        const statusDiv = document.createElement('div');
        statusDiv.id = 'gesture-status';
        statusDiv.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 12px; height: 12px; background: #00ff00; border-radius: 50%; animation: pulse 2s infinite;"></div>
                <span>Gesture Control Active</span>
            </div>
            <div id="gesture-text" style="margin-top: 5px; font-size: 18px; font-weight: bold;"></div>
        `;
        
        this.overlay.appendChild(statusDiv);
        document.body.appendChild(this.overlay);
        
        // Add CSS animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.3; }
            }
        `;
        document.head.appendChild(style);
    }
    
    updateOverlayStyle() {
        if (!this.overlay) return;
        
        const bgColor = this.highContrast ? 'rgba(0, 0, 0, 0.9)' : 'rgba(0, 0, 0, 0.7)';
        const textColor = this.highContrast ? '#ffffff' : '#ffffff';
        
        this.overlay.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            background: ${bgColor};
            color: ${textColor};
            border-radius: 10px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            z-index: 999999;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border: ${this.highContrast ? '2px solid #ffffff' : 'none'};
        `;
    }
    
    removeOverlay() {
        if (this.overlay && this.overlay.parentNode) {
            this.overlay.parentNode.removeChild(this.overlay);
            this.overlay = null;
        }
    }
    
    handleGestureData(data) {
        if (data.type === 'text') {
            this.handleTextInput(data.text);
        } else if (data.type === 'search') {
            this.performSearch(data.query);
        } else if (data.type === 'keyboard_toggle') {
            this.keyboardVisible = data.visible;
        }
        
        // Update overlay with current text
        if (this.overlay) {
            const textDiv = this.overlay.querySelector('#gesture-text');
            if (textDiv && data.text) {
                textDiv.textContent = data.text;
            }
        }
    }
    
    handleTextInput(text) {
        // Find active input element
        const activeElement = document.activeElement;
        
        if (activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA')) {
            // Insert text at cursor position
            const start = activeElement.selectionStart;
            const end = activeElement.selectionEnd;
            const currentValue = activeElement.value;
            
            activeElement.value = currentValue.substring(0, start) + text + currentValue.substring(end);
            activeElement.selectionStart = activeElement.selectionEnd = start + text.length;
            
            // Trigger input event
            activeElement.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }
    
    performSearch(query) {
        if (!query) {
            return;
        }
        
        // Build search URL based on platform
        let searchUrl;
        switch (this.platform) {
            case 'google':
                searchUrl = `https://www.google.com/search?q=${encodeURIComponent(query)}`;
                break;
            case 'youtube':
                searchUrl = `https://www.youtube.com/results?search_query=${encodeURIComponent(query)}`;
                break;
            case 'instagram':
                searchUrl = `https://www.instagram.com/explore/tags/${encodeURIComponent(query)}`;
                break;
            case 'browser':
                searchUrl = `https://www.bing.com/search?q=${encodeURIComponent(query)}`;
                break;
            default:
                searchUrl = `https://www.google.com/search?q=${encodeURIComponent(query)}`;
        }
        
        // Navigate to search URL
        window.location.href = searchUrl;
        
        if (this.voiceFeedback) {
            this.speak(`Searching ${this.platform} for ${query}`);
        }
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 15px 25px;
            background: ${type === 'success' ? '#4CAF50' : type === 'error' ? '#f44336' : '#2196F3'};
            color: white;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            z-index: 1000000;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            animation: slideDown 0.3s ease-out;
        `;
        notification.textContent = message;
        
        // Add animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideDown {
                from {
                    opacity: 0;
                    transform: translateX(-50%) translateY(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateX(-50%) translateY(0);
                }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideDown 0.3s ease-out reverse';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
    
    speak(text) {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(utterance);
        }
    }
}

// Initialize extension
const gestureKeyboard = new GestureKeyboardExtension();

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GestureKeyboardExtension;
}
