/* ============================================
   THEME.JS - Dark/Light Mode Management
   ============================================ */

const Theme = {
    STORAGE_KEY: 'nephroscan-theme',
    
    /**
     * Initialize theme from localStorage or system preference
     */
    init() {
        const savedTheme = localStorage.getItem(this.STORAGE_KEY);
        
        if (savedTheme) {
            this.set(savedTheme);
        } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            this.set('dark');
        } else {
            this.set('light');
        }
        
        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (!localStorage.getItem(this.STORAGE_KEY)) {
                this.set(e.matches ? 'dark' : 'light');
            }
        });
        
        // Setup toggle listener
        this.setupToggle();
    },
    
    /**
     * Set theme and update DOM
     */
    set(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(this.STORAGE_KEY, theme);
        
        // Update meta theme-color for mobile browsers
        const metaTheme = document.querySelector('meta[name="theme-color"]');
        if (metaTheme) {
            metaTheme.content = theme === 'dark' ? '#0f1415' : '#f8fafa';
        }
    },
    
    /**
     * Get current theme
     */
    get() {
        return document.documentElement.getAttribute('data-theme') || 'light';
    },
    
    /**
     * Toggle between light and dark
     */
    toggle() {
        const current = this.get();
        const next = current === 'dark' ? 'light' : 'dark';
        this.set(next);
        
        // Add animation class to body
        document.body.classList.add('theme-transitioning');
        setTimeout(() => {
            document.body.classList.remove('theme-transitioning');
        }, 300);
    },
    
    /**
     * Setup click listener for theme switch
     */
    setupToggle() {
        const toggle = document.getElementById('theme-switch');
        if (toggle) {
            toggle.addEventListener('click', () => this.toggle());
        }
    }
};

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => Theme.init());