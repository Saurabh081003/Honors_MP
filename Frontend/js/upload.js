/* ============================================
   UPLOAD.JS - File Upload Management
   ============================================ */

const Upload = {
    maxFileSize: 10 * 1024 * 1024, // 10MB
    allowedTypes: ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'],
    selectedFile: null,
    
    // DOM Elements
    elements: {},
    
    /**
     * Initialize upload functionality
     */
    init() {
        this.cacheElements();
        this.bindEvents();
    },
    
    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements = {
            uploadZone: document.getElementById('upload-zone'),
            fileInput: document.getElementById('file-input'),
            previewZone: document.getElementById('preview-zone'),
            previewImage: document.getElementById('preview-image'),
            previewFilename: document.getElementById('preview-filename'),
            previewClose: document.getElementById('preview-close'),
            resetBtn: document.getElementById('reset-btn'),
            classifyBtn: document.getElementById('classify-btn'),
            previewOverlay: document.getElementById('preview-overlay')
        };
    },
    
    /**
     * Bind event listeners
     */
    bindEvents() {
        const { uploadZone, fileInput, previewClose, resetBtn, classifyBtn } = this.elements;
        
        // Click to browse
        uploadZone.addEventListener('click', () => fileInput.click());
        
        // File input change
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files));
        
        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadZone.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadZone.addEventListener('drop', (e) => this.handleDrop(e));
        
        // Preview actions
        previewClose.addEventListener('click', () => this.reset());
        resetBtn.addEventListener('click', () => this.reset());
        classifyBtn.addEventListener('click', () => this.classify());
        
        // Prevent default drag behaviors on window
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
            document.body.addEventListener(event, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
    },
    
    /**
     * Handle drag over event
     */
    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        this.elements.uploadZone.classList.add('drag-over');
    },
    
    /**
     * Handle drag leave event
     */
    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        this.elements.uploadZone.classList.remove('drag-over');
    },
    
    /**
     * Handle file drop
     */
    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        this.elements.uploadZone.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFileSelect(files);
        }
    },
    
    /**
     * Handle file selection
     */
    handleFileSelect(files) {
        if (!files || files.length === 0) return;
        
        const file = files[0];
        
        // Validate file type
        if (!this.allowedTypes.includes(file.type)) {
            Toast.show('Please select a valid image file (JPG, PNG)', 'error');
            return;
        }
        
        // Validate file size
        if (file.size > this.maxFileSize) {
            Toast.show('File size must be less than 10MB', 'error');
            return;
        }
        
        this.selectedFile = file;
        this.showPreview(file);
    },
    
    /**
     * Validate if image looks like a CT scan
     */
    validateCTImage(img) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Use smaller size for faster processing
            const size = 100;
            canvas.width = size;
            canvas.height = size;
            
            ctx.drawImage(img, 0, 0, size, size);
            const imageData = ctx.getImageData(0, 0, size, size);
            const data = imageData.data;
            
            let grayscalePixels = 0;
            let totalPixels = data.length / 4;
            let darkPixels = 0;
            
            for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                
                // Check if pixel is grayscale (R ≈ G ≈ B)
                const maxDiff = Math.max(
                    Math.abs(r - g),
                    Math.abs(g - b),
                    Math.abs(r - b)
                );
                
                if (maxDiff < 30) {
                    grayscalePixels++;
                }
                
                // Check for dark pixels (common in CT)
                const brightness = (r + g + b) / 3;
                if (brightness < 100) {
                    darkPixels++;
                }
            }
            
            const grayscaleRatio = grayscalePixels / totalPixels;
            const darkRatio = darkPixels / totalPixels;
            
            // CT scans are typically:
            // - Mostly grayscale (>70% grayscale pixels)
            // - Have significant dark regions (>20% dark pixels)
            const looksLikeCT = grayscaleRatio > 0.7 && darkRatio > 0.15;
            
            resolve({
                isValid: looksLikeCT,
                grayscaleRatio: grayscaleRatio,
                darkRatio: darkRatio
            });
        });
    },
    
    /**
     * Show image preview
     */
    showPreview(file) {
        const reader = new FileReader();
        
        reader.onload = async (e) => {
            const imgSrc = e.target.result;
            
            // Create temp image to validate
            const tempImg = new Image();
            tempImg.onload = async () => {
                // Validate CT image
                const validation = await this.validateCTImage(tempImg);
                
                if (!validation.isValid) {
                    Toast.show(
                        'This doesn\'t look like a CT scan. Results may be inaccurate.',
                        'warning'
                    );
                }
                
                // Show preview regardless (with warning)
                this.elements.previewImage.src = imgSrc;
                this.elements.previewFilename.textContent = file.name;
                
                // Hide upload zone, show preview
                this.elements.uploadZone.classList.add('hidden');
                this.elements.previewZone.classList.remove('hidden');
                
                // Hide results if showing
                Results.hide();
            };
            tempImg.src = imgSrc;
        };
        
        reader.onerror = () => {
            Toast.show('Error reading file', 'error');
        };
        
        reader.readAsDataURL(file);
    },
    
    /**
     * Reset upload state
     */
    reset() {
        this.selectedFile = null;
        this.elements.fileInput.value = '';
        this.elements.previewImage.src = '';
        this.elements.previewFilename.textContent = '';
        
        // Show upload zone, hide preview
        this.elements.uploadZone.classList.remove('hidden');
        this.elements.previewZone.classList.add('hidden');
        
        // Hide results
        Results.hide();
    },
    
    /**
     * Get selected file
     */
    getFile() {
        return this.selectedFile;
    },
    
    /**
     * Trigger classification
     */
    async classify() {
        if (!this.selectedFile) {
            Toast.show('Please select an image first', 'warning');
            return;
        }
        
        // Show loading state
        this.setLoading(true);
        
        try {
            const result = await API.predict(this.selectedFile);
            
            if (result.success) {
                Results.show(result);
                Toast.show('Analysis complete!', 'success');
            } else {
                Toast.show(result.error || 'Classification failed', 'error');
            }
        } catch (error) {
            console.error('Classification error:', error);
            Toast.show('Failed to connect to server', 'error');
        } finally {
            this.setLoading(false);
        }
    },
    
    /**
     * Set loading state
     */
    setLoading(loading) {
        const { classifyBtn, previewOverlay } = this.elements;
        
        if (loading) {
            classifyBtn.disabled = true;
            classifyBtn.innerHTML = `
                <span class="material-icons-round">hourglass_empty</span>
                Analyzing...
            `;
            previewOverlay.classList.add('visible');
        } else {
            classifyBtn.disabled = false;
            classifyBtn.innerHTML = `
                <span class="material-icons-round">psychology</span>
                Analyze
            `;
            previewOverlay.classList.remove('visible');
        }
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => Upload.init());