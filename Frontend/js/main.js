/* ============================================
   MAIN.JS - Main Application Entry Point
   ============================================ */

/* === Toast Notification System === */
const Toast = {
    container: null,
    
    /**
     * Initialize toast container
     */
    init() {
        this.container = document.getElementById('toast-container');
    },
    
    /**
     * Show a toast notification
     */
    show(message, type = 'info', duration = 4000) {
        if (!this.container) this.init();
        
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const icons = {
            success: 'check_circle',
            error: 'error',
            warning: 'warning',
            info: 'info'
        };
        
        toast.innerHTML = `
            <span class="material-icons-round">${icons[type] || icons.info}</span>
            <span>${message}</span>
        `;
        
        this.container.appendChild(toast);
        
        // Auto remove after duration
        setTimeout(() => {
            toast.classList.add('toast-out');
            setTimeout(() => toast.remove(), 200);
        }, duration);
        
        return toast;
    }
};

/* === Modal System === */
const Modal = {
    overlay: null,
    activeModal: null,
    
    /**
     * Initialize modal system
     */
    init() {
        this.overlay = document.getElementById('modal-overlay');
        
        // Close button
        const closeBtn = document.getElementById('modal-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.close());
        }
        
        // Click outside to close
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) {
                this.close();
            }
        });
        
        // Escape key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen()) {
                this.close();
            }
        });
        
        // Info button
        const infoBtn = document.getElementById('info-btn');
        if (infoBtn) {
            infoBtn.addEventListener('click', () => this.open('info-modal'));
        }
    },
    
    /**
     * Open a modal
     */
    open(modalId) {
        this.overlay.classList.remove('hidden');
        
        requestAnimationFrame(() => {
            this.overlay.classList.add('visible');
        });
        
        this.activeModal = modalId;
        document.body.style.overflow = 'hidden';
    },
    
    /**
     * Close the modal
     */
    close() {
        this.overlay.classList.remove('visible');
        
        setTimeout(() => {
            this.overlay.classList.add('hidden');
        }, 300);
        
        this.activeModal = null;
        document.body.style.overflow = '';
    },
    
    /**
     * Check if modal is open
     */
    isOpen() {
        return this.activeModal !== null;
    }
};

/* === Ripple Effect === */
const Ripple = {
    /**
     * Initialize ripple effects on buttons
     */
    init() {
        document.addEventListener('click', (e) => {
            const btn = e.target.closest('.btn-filled');
            if (!btn) return;
            
            const rect = btn.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const ripple = document.createElement('span');
            ripple.className = 'btn-ripple';
            ripple.style.left = `${x}px`;
            ripple.style.top = `${y}px`;
            
            btn.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    }
};

/* === Intersection Observer for Animations === */
const AnimationObserver = {
    init() {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('visible');
                        observer.unobserve(entry.target);
                    }
                });
            },
            {
                threshold: 0.1,
                rootMargin: '0px 0px -50px 0px'
            }
        );
        
        document.querySelectorAll('.animate-on-scroll').forEach(el => {
            observer.observe(el);
        });
    }
};

/* === Keyboard Shortcuts === */
const Shortcuts = {
    init() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + U: Upload
            if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
                e.preventDefault();
                document.getElementById('file-input')?.click();
            }
            
            // Ctrl/Cmd + Enter: Classify
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                document.getElementById('classify-btn')?.click();
            }
            
            // Ctrl/Cmd + D: Toggle theme
            if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
                e.preventDefault();
                Theme.toggle();
            }
        });
    }
};

/* === Service Worker Registration (PWA Ready) === */
const PWA = {
    async init() {
        if ('serviceWorker' in navigator) {
            try {
                // Uncomment when you have a service worker
                // await navigator.serviceWorker.register('/sw.js');
                // console.log('Service Worker registered');
            } catch (error) {
                // console.log('Service Worker registration failed:', error);
            }
        }
    }
};

/* === App Initialization === */
const App = {
    /**
     * Initialize the application
     */
    init() {
        console.log('%c🫘 NephroScan AI', 'font-size: 24px; font-weight: bold; color: #00897b;');
        console.log('%cKidney CT Classification System', 'font-size: 12px; color: #666;');
        
        // Initialize all modules
        Toast.init();
        Modal.init();
        Ripple.init();
        AnimationObserver.init();
        Shortcuts.init();
        PWA.init();
        
        // Log keyboard shortcuts
        console.log('\n%cKeyboard Shortcuts:', 'font-weight: bold;');
        console.log('  Ctrl/Cmd + U  →  Upload image');
        console.log('  Ctrl/Cmd + Enter  →  Analyze');
        console.log('  Ctrl/Cmd + D  →  Toggle theme');
        console.log('  Escape  →  Close modal');
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => App.init());

// Expose modules globally for debugging
window.App = App;
window.Toast = Toast;
window.Modal = Modal;
window.Theme = Theme;
window.Upload = Upload;
window.Results = Results;
window.API = API;