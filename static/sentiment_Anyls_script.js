// Theme management
const THEME_KEY = "sentiment-analysis-theme";

function initTheme() {
  const savedTheme = localStorage.getItem(THEME_KEY) || "light";
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme =
    savedTheme === "auto" ? (prefersDark ? "dark" : "light") : savedTheme;

  setTheme(theme);
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const themeIcon = document.getElementById("theme-icon");

  if (theme === "dark") {
    themeIcon.textContent = "☀️";
    themeIcon.parentElement.title = "Switch to light mode";
  } else {
    themeIcon.textContent = "🌙";
    themeIcon.parentElement.title = "Switch to dark mode";
  }

  localStorage.setItem(THEME_KEY, theme);
}

function toggleTheme() {
  const currentTheme =
    document.documentElement.getAttribute("data-theme") || "light";
  const newTheme = currentTheme === "light" ? "dark" : "light";
  setTheme(newTheme);
}

// Initialize theme on page load
document.addEventListener("DOMContentLoaded", function () {
  initTheme();

  // Add animation delays for staggered appearance
  const cards = document.querySelectorAll(".result-card");
  cards.forEach((card, index) => {
    card.style.animationDelay = `${index * 0.2}s`;
  });
});

// Listen for system theme changes
window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", (e) => {
    const currentTheme = localStorage.getItem(THEME_KEY);
    if (currentTheme === "auto" || !currentTheme) {
      setTheme(e.matches ? "dark" : "light");
    }
  });

// Function to copy results to clipboard
function copyResults() {
  const btn = document.getElementById("copyBtn");
  const sentiment = "{{ sentiment }}";
  const confidence = '{{ "%.1f"|format(confidence) }}';
  const text = "{{ summary_text }}";

  const results = `Sentiment Analysis Results:
Text: "${text}"
Sentiment: ${sentiment}
Confidence: ${confidence}%`;

  navigator.clipboard
    .writeText(results)
    .then(() => {
      const originalText = btn.innerHTML;
      btn.innerHTML = "✅ Copied!";
      btn.style.background = "linear-gradient(135deg, #10b981, #059669)";
      btn.style.color = "white";

      setTimeout(() => {
        btn.innerHTML = originalText;
        btn.style.background = "";
        btn.style.color = "";
      }, 2000);
    })
    .catch(() => {
      // Fallback for older browsers
      const textArea = document.createElement("textarea");
      textArea.value = results;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);

      const originalText = btn.innerHTML;
      btn.innerHTML = "✅ Copied!";
      setTimeout(() => {
        btn.innerHTML = originalText;
      }, 2000);
    });
}

// Function to share results (using Web Share API if available)
function shareResults() {
  const sentiment = "{{ sentiment }}";
  const confidence = '{{ "%.1f"|format(confidence) }}';
  const text = "{{ summary_text }}";

  const shareData = {
    title: "Sentiment Analysis Results",
    text: `Sentiment: ${sentiment} (${confidence}% confidence)\nText: "${text}"`,
    url: window.location.href,
  };

  if (navigator.share) {
    navigator.share(shareData);
  } else {
    // Fallback - copy to clipboard
    copyResults();
  }
}

// Add hover effects for sentiment cards
document.querySelectorAll(".sentiment-card").forEach((card) => {
  card.addEventListener("mouseenter", function () {
    this.style.transform = "translateY(-5px) scale(1.02)";
  });

  card.addEventListener("mouseleave", function () {
    this.style.transform = "translateY(0) scale(1)";
  });
});
