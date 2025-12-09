/* ============================================
   API.JS - Backend Communication
   ============================================ */

const API = {
    // Base URL for API - change this in production
    baseURL: 'http://localhost:5000',
    
    /**
     * Health check endpoint
     */
    async health() {
        try {
            const response = await fetch(`${this.baseURL}/api/health`);
            return await response.json();
        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'error', error: error.message };
        }
    },
    
    /**
     * Predict/classify an image
     */
    async predict(file) {
        try {
            const formData = new FormData();
            formData.append('image', file);
            
            const response = await fetch(`${this.baseURL}/api/predict`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Server error');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Prediction failed:', error);
            return { success: false, error: error.message };
        }
    },
    
    /**
     * Get class information
     */
    async getClasses() {
        try {
            const response = await fetch(`${this.baseURL}/api/classes`);
            return await response.json();
        } catch (error) {
            console.error('Failed to get classes:', error);
            return null;
        }
    },
    
    /**
     * Check if API is available
     */
    async isAvailable() {
        try {
            const result = await this.health();
            return result.status === 'healthy';
        } catch {
            return false;
        }
    }
};

// Check API availability on load
document.addEventListener('DOMContentLoaded', async () => {
    const available = await API.isAvailable();
    
    if (!available) {
        Toast.show('Backend server is not running. Start the Flask server first.', 'warning');
    }
});