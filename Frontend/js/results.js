/* ============================================
   RESULTS.JS - Results Display Management
   ============================================ */

const Results = {
    // DOM Elements
    elements: {},
    
    // Icon mapping for each class
    iconMap: {
        'Cyst': 'water_drop',
        'Normal': 'check_circle',
        'Stone': 'hexagon',
        'Tumor': 'warning'
    },
    
    // Color classes for result styling
    colorMap: {
        'Cyst': 'cyst',
        'Normal': 'normal',
        'Stone': 'stone',
        'Tumor': 'tumor'
    },
    
    /**
     * Initialize results module
     */
    init() {
        this.cacheElements();
        this.setupAccordion();
    },
    
    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements = {
            resultsCard: document.getElementById('results-card'),
            resultPrimary: document.getElementById('result-primary'),
            resultBadge: document.getElementById('result-badge'),
            resultIcon: document.getElementById('result-icon'),
            resultClass: document.getElementById('result-class'),
            confidenceFill: document.getElementById('confidence-fill'),
            confidenceValue: document.getElementById('confidence-value'),
            probabilityBars: document.getElementById('probability-bars'),
            explanationTrigger: document.getElementById('explanation-trigger'),
            explanationContent: document.getElementById('explanation-content'),
            explanationDesc: document.getElementById('explanation-desc'),
            characteristicsList: document.getElementById('characteristics-list'),
            recommendationText: document.getElementById('recommendation-text'),
            heatmapSection: document.getElementById('heatmap-section'),
            heatmapImage: document.getElementById('heatmap-image')
        };
    },
    
    /**
     * Setup accordion functionality
     */
    setupAccordion() {
        const { explanationTrigger, explanationContent } = this.elements;
        
        explanationTrigger.addEventListener('click', () => {
            explanationTrigger.classList.toggle('active');
            explanationContent.classList.toggle('open');
        });
    },
    
    /**
     * Show results
     */
    show(data) {
        const { prediction, explanation, heatmap } = data;
        
        // Update primary result
        this.updatePrimaryResult(prediction);
        
        // Update probability bars
        this.updateProbabilities(prediction);
        
        // Update explanation
        this.updateExplanation(explanation);
        
        // Update heatmap
        this.updateHeatmap(heatmap);
        
        // Show results card with animation
        this.elements.resultsCard.classList.remove('hidden');
        
        // Scroll to results on mobile
        if (window.innerWidth <= 1024) {
            setTimeout(() => {
                this.elements.resultsCard.scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'start'
                });
            }, 100);
        }
    },
    
    /**
     * Hide results
     */
    hide() {
        this.elements.resultsCard.classList.add('hidden');
        
        // Close accordion
        this.elements.explanationTrigger.classList.remove('active');
        this.elements.explanationContent.classList.remove('open');
    },
    
    /**
     * Update primary result display
     */
    updatePrimaryResult(prediction) {
        const { resultIcon, resultClass, confidenceFill, confidenceValue } = this.elements;
        
        // Update icon
        resultIcon.textContent = this.iconMap[prediction.class] || 'science';
        
        // Update class name
        resultClass.textContent = prediction.class;
        
        // Animate confidence bar
        setTimeout(() => {
            confidenceFill.style.width = `${prediction.confidence}%`;
        }, 100);
        
        // Update confidence value with animation
        this.animateNumber(confidenceValue, 0, prediction.confidence, 600, '%');
    },
    
    /**
     * Update probability bars
     */
    updateProbabilities(prediction) {
        const { probabilityBars } = this.elements;
        const { probabilities } = prediction;
        
        // Sort probabilities by value (descending)
        const sorted = Object.entries(probabilities)
            .sort((a, b) => b[1] - a[1]);
        
        // Clear existing bars
        probabilityBars.innerHTML = '';
        
        // Create bars
        sorted.forEach(([className, prob], index) => {
            const isHighest = className === prediction.class;
            
            const item = document.createElement('div');
            item.className = `prob-item stagger-item ${isHighest ? 'highlight' : ''}`;
            item.setAttribute('data-class', className);
            item.style.animationDelay = `${index * 50}ms`;
            
            item.innerHTML = `
                <div class="prob-header">
                    <span class="prob-label">${className}</span>
                    <span class="prob-value">0%</span>
                </div>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: 0%"></div>
                </div>
            `;
            
            probabilityBars.appendChild(item);
            
            // Animate bar after a short delay
            setTimeout(() => {
                const fill = item.querySelector('.prob-fill');
                const value = item.querySelector('.prob-value');
                
                fill.style.width = `${prob}%`;
                this.animateNumber(value, 0, prob, 400, '%');
            }, 200 + (index * 100));
        });
    },
    
    /**
     * Update explanation section
     */
    updateExplanation(explanation) {
        const { explanationDesc, characteristicsList, recommendationText } = this.elements;
        
        // Update description
        explanationDesc.textContent = explanation.description || 'No description available.';
        
        // Update characteristics
        characteristicsList.innerHTML = '';
        
        if (explanation.characteristics && explanation.characteristics.length > 0) {
            explanation.characteristics.forEach(char => {
                const li = document.createElement('li');
                li.textContent = char;
                characteristicsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No specific characteristics available.';
            characteristicsList.appendChild(li);
        }
        
        // Update recommendation
        recommendationText.textContent = explanation.recommendation || 'Consult a healthcare professional for proper evaluation.';
        
        // Auto-open accordion
        this.elements.explanationTrigger.classList.add('active');
        this.elements.explanationContent.classList.add('open');
    },
    
    /**
     * Update heatmap section
     */
    updateHeatmap(heatmapData) {
        const { heatmapSection, heatmapImage } = this.elements;
        
        if (heatmapData) {
            heatmapImage.src = `data:image/png;base64,${heatmapData}`;
            heatmapSection.classList.remove('hidden');
        } else {
            heatmapSection.classList.add('hidden');
        }
    },
    
    /**
     * Animate number from start to end
     */
    animateNumber(element, start, end, duration, suffix = '') {
        const startTime = performance.now();
        
        const update = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function (ease-out-cubic)
            const eased = 1 - Math.pow(1 - progress, 3);
            
            const current = start + (end - start) * eased;
            element.textContent = `${current.toFixed(1)}${suffix}`;
            
            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.textContent = `${end.toFixed(1)}${suffix}`;
            }
        };
        
        requestAnimationFrame(update);
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => Results.init());