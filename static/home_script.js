        // Theme Management System
        class ThemeManager {
            constructor() {
                this.themeToggle = document.getElementById('themeToggle');
                this.init();
            }

            init() {
                // Load saved theme or default to light
                const savedTheme = this.getSavedTheme();
                this.setTheme(savedTheme);
                
                // Set toggle position
                this.themeToggle.checked = savedTheme === 'dark';
                
                // Add event listener
                this.themeToggle.addEventListener('change', () => {
                    const newTheme = this.themeToggle.checked ? 'dark' : 'light';
                    this.setTheme(newTheme);
                });
            }

            getSavedTheme() {
                // Try to get from URL parameters first (for cross-page consistency)
                const urlParams = new URLSearchParams(window.location.search);
                const urlTheme = urlParams.get('theme');
                
                if (urlTheme && (urlTheme === 'light' || urlTheme === 'dark')) {
                    this.saveTheme(urlTheme);
                    return urlTheme;
                }

                // Fallback to stored preference or system preference
                const stored = localStorage.getItem('news-analyzer-theme');
                if (stored) {
                    return stored;
                }

                // Default to system preference or light
                return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
            }

            setTheme(theme) {
                document.documentElement.setAttribute('data-theme', theme);
                this.saveTheme(theme);
                
                // Add theme parameter to all internal links
                this.updateInternalLinks(theme);
            }

            saveTheme(theme) {
                localStorage.setItem('news-analyzer-theme', theme);
            }

            updateInternalLinks(theme) {
                // Update all internal links to include theme parameter
                const internalLinks = document.querySelectorAll('a[href^="/"], a[href^="./"], a[href^="../"]');
                internalLinks.forEach(link => {
                    const url = new URL(link.href, window.location.origin);
                    url.searchParams.set('theme', theme);
                    link.href = url.toString();
                });
            }
        }

        // Initialize theme management
        const themeManager = new ThemeManager();

        // Add smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Animate stats on scroll
        const observerOptions = {
            threshold: 0.5,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.animationPlayState = 'running';
                }
            });
        }, observerOptions);

        document.querySelectorAll('.stat-number').forEach(stat => {
            observer.observe(stat);
        });

        // Add loading state for CTA button
        document.querySelector('.cta-primary').addEventListener('click', function(e) {
            const btn = this;
            const originalText = btn.innerHTML;
            btn.innerHTML = '⏳ Loading...';
            btn.style.pointerEvents = 'none';
            
            // Restore button after a short delay (in case navigation is slow)
            setTimeout(() => {
                btn.innerHTML = originalText;
                btn.style.pointerEvents = 'auto';
            }, 3000);
        });

        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            // Only update if user hasn't manually set a preference
            if (!localStorage.getItem('news-analyzer-theme')) {
                const newTheme = e.matches ? 'dark' : 'light';
                themeManager.setTheme(newTheme);
                themeManager.themeToggle.checked = newTheme === 'dark';
            }
        });
