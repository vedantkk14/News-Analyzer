// Theme Management System - Default to light mode
class ThemeManager {
  constructor() {
    this.themeToggle = document.getElementById("themeToggle");
    this.init();
  }

  init() {
    // Always default to light mode
    const defaultTheme = "light";
    this.setTheme(defaultTheme);

    // Set toggle position
    this.themeToggle.checked = false;

    // Add event listener
    this.themeToggle.addEventListener("change", () => {
      const newTheme = this.themeToggle.checked ? "dark" : "light";
      this.setTheme(newTheme);
    });
  }

  setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    this.saveTheme(theme);

    // Add theme parameter to all internal links
    this.updateInternalLinks(theme);
  }

  saveTheme(theme) {
    localStorage.setItem("news-analyzer-theme", theme);
  }

  updateInternalLinks(theme) {
    // Update all internal links to include theme parameter
    const internalLinks = document.querySelectorAll(
      'a[href^="/"], a[href^="./"], a[href^="../"]'
    );
    internalLinks.forEach((link) => {
      const url = new URL(link.href, window.location.origin);
      url.searchParams.set("theme", theme);
      link.href = url.toString();
    });
  }
}

// Initialize theme management
const themeManager = new ThemeManager();

// Get DOM elements
const inputText = document.getElementById("inputText");
const charCount = document.getElementById("charCount");
const resetBtn = document.getElementById("resetBtn");
const summarizeBtn = document.getElementById("summarizeBtn");
const outputArea = document.getElementById("outputArea");
const modelOptions = document.querySelectorAll(".model-option");

let currentSummary = "";
let currentSpeech = null;
let selectedModel = "default";
let originalTextLength = 0;

// Check for TTS support
const ttsSupported = "speechSynthesis" in window;

// Model selection handler
modelOptions.forEach((option) => {
  option.addEventListener("click", () => {
    // Update active state
    modelOptions.forEach((opt) => opt.classList.remove("active"));
    option.classList.add("active");

    // Update selected model
    selectedModel = option.dataset.model;

    // If there's existing text, re-summarize with new model
    if (inputText.value.trim()) {
      summarizeBtn.click();
    }
  });
});

// Text summarization function
async function summarizeText(text, model = "default") {
  try {
    const endpoint = model === "pegasus" ? "/summarize_pegasus" : "/summarize";
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    });

    const data = await response.json();
    return data.summary || data.error || "Unable to generate summary";
  } catch (error) {
    // Fallback to local summarization if API fails
    return localSummarize(text, model);
  }
}

// Local fallback summarization
function localSummarize(text, model = "default") {
  if (!text.trim()) return "";

  // Split into sentences
  const sentences = text.match(/[^\.!?]+[\.!?]+/g) || [text];

  if (model === "pegasus") {
    // For Pegasus model simulation - return just one sentence
    if (sentences.length === 0) return text.substring(0, 100) + "...";

    // Score sentences and return the highest scoring one
    const words = text
      .toLowerCase()
      .replace(/[^\w\s]/g, "")
      .split(/\s+/)
      .filter((word) => word.length > 3);

    const wordFreq = {};
    words.forEach((word) => {
      wordFreq[word] = (wordFreq[word] || 0) + 1;
    });

    const scoredSentences = sentences.map((sentence, index) => {
      const sentenceWords = sentence
        .toLowerCase()
        .replace(/[^\w\s]/g, "")
        .split(/\s+/);

      const score =
        sentenceWords.reduce((acc, word) => {
          return acc + (wordFreq[word] || 0);
        }, 0) / sentenceWords.length;

      // Boost score for earlier sentences
      const positionBoost = 1 - (index / sentences.length) * 0.3;

      return {
        sentence: sentence.trim(),
        score: score * positionBoost,
      };
    });

    return scoredSentences.sort((a, b) => b.score - a.score)[0].sentence;
  } else {
    // Default model - return 1-2 sentences
    if (sentences.length <= 2) {
      return sentences.join(" ").trim();
    }

    // Score sentences based on word frequency and position
    const words = text
      .toLowerCase()
      .replace(/[^\w\s]/g, "")
      .split(/\s+/)
      .filter((word) => word.length > 3);

    const wordFreq = {};
    words.forEach((word) => {
      wordFreq[word] = (wordFreq[word] || 0) + 1;
    });

    const scoredSentences = sentences.map((sentence, index) => {
      const sentenceWords = sentence
        .toLowerCase()
        .replace(/[^\w\s]/g, "")
        .split(/\s+/);

      const score =
        sentenceWords.reduce((acc, word) => {
          return acc + (wordFreq[word] || 0);
        }, 0) / sentenceWords.length;

      // Boost score for earlier sentences
      const positionBoost = 1 - (index / sentences.length) * 0.3;

      return {
        sentence: sentence.trim(),
        score: score * positionBoost,
        index,
      };
    });

    // Get top 2 sentences
    const topSentences = scoredSentences
      .sort((a, b) => b.score - a.score)
      .slice(0, 2)
      .sort((a, b) => a.index - b.index)
      .map((item) => item.sentence);

    return topSentences.join(" ");
  }
}

// Update character count and button states
function updateUI() {
  const textLength = inputText.value.length;
  originalTextLength = textLength;
  charCount.textContent = textLength.toLocaleString() + " characters";

  // Add active class for character count animation
  if (textLength > 0) {
    charCount.classList.add("active");
  } else {
    charCount.classList.remove("active");
  }

  const hasText = inputText.value.trim().length > 0;
  resetBtn.disabled = !hasText;
  summarizeBtn.disabled = !hasText;
}

// Show loading state
function showLoading() {
  outputArea.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p style="color: var(--card-secondary); font-weight: 500;">Analyzing text...</p>
                </div>
            `;
}

// Show summary result
function showSummary(summary) {
  currentSummary = summary;

  // Calculate stats
  const summaryLength = summary.length;
  const reduction = originalTextLength - summaryLength;
  const reductionPercentage =
    originalTextLength > 0
      ? Math.round((reduction / originalTextLength) * 100)
      : 0;

  const escapedSummary = summary.replace(/'/g, "\\'").replace(/"/g, "&quot;");

  let ttsWarning = "";
  if (!ttsSupported) {
    ttsWarning = `
                    <div class="tts-warning">
                        ⚠ Text-to-Speech is not supported in your browser. The Listen button will be disabled.
                    </div>
                `;
  }

  outputArea.innerHTML = `
                <div style="width: 100%;">
                    ${ttsWarning}
                    <div class="summary-box">
                        <div class="summary-content">
                            <p class="summary-text">${summary}</p>
                        </div>
                        <div class="summary-stats">
                            <div class="stat-item">
                                <span class="stat-value">${summaryLength.toLocaleString()}</span>
                                <div class="stat-label">Characters</div>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value">-${reduction.toLocaleString()}</span>
                                <div class="stat-label">Reduced</div>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value">${reductionPercentage}%</span>
                                <div class="stat-label">Compression</div>
                            </div>
                        </div>
                    </div>
                    <div class="action-buttons">
                        <button onclick="copySummary('${escapedSummary}')" class="btn btn-copy">
                            📋 Copy Summary
                        </button>
                        <button onclick="analyzeSentiment('${escapedSummary}')" class="btn btn-sentiment">
                            💭 Analyze Sentiment
                        </button>
                        <button onclick="listenToSummary('${escapedSummary}')" class="btn btn-listen" ${
    !ttsSupported ? "disabled" : ""
  }>
                            🔊 Listen
                        </button>
                    </div>
                </div>
            `;
}

// Show error state
function showError(error) {
  outputArea.innerHTML = `
                <div style="width: 100%; text-align: center;">
                    <div style="color: #e53e3e; margin-bottom: 1rem;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem; animation: shake 0.5s ease-in-out;">❌</div>
                        <p>Error: ${error}</p>
                    </div>
                    <button onclick="showEmptyState()" class="btn btn-secondary">
                        🔄 Try Again
                    </button>
                </div>
            `;
}

// Show empty state
function showEmptyState() {
  currentSummary = "";
  originalTextLength = 0;
  // Stop any ongoing speech
  if (currentSpeech) {
    speechSynthesis.cancel();
    currentSpeech = null;
  }
  outputArea.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📄</div>
                    <p>Your summary will appear here</p>
                </div>
            `;
}

// Copy summary to clipboard
function copySummary(text) {
  const btn = event.target;

  navigator.clipboard
    .writeText(text)
    .then(() => {
      // Show success animation
      btn.classList.add("copied");
      const originalText = btn.innerHTML;
      btn.innerHTML = "✅ Copied!";

      setTimeout(() => {
        btn.innerHTML = originalText;
        btn.classList.remove("copied");
      }, 2000);
    })
    .catch(() => {
      // Fallback for older browsers
      const textArea = document.createElement("textarea");
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);

      btn.classList.add("copied");
      const originalText = btn.innerHTML;
      btn.innerHTML = "✅ Copied!";

      setTimeout(() => {
        btn.innerHTML = originalText;
        btn.classList.remove("copied");
      }, 2000);
    });
}

// Text-to-Speech function
function listenToSummary(summaryText) {
  if (!ttsSupported) {
    alert("Text-to-Speech is not supported in your browser.");
    return;
  }

  const btn = event.target;

  // If already speaking, stop
  if (speechSynthesis.speaking) {
    speechSynthesis.cancel();
    btn.classList.remove("listening");
    btn.innerHTML = "🔊 Listen";
    currentSpeech = null;
    return;
  }

  // Create speech synthesis utterance
  const utterance = new SpeechSynthesisUtterance(summaryText);

  // Configure speech settings
  utterance.rate = 0.9; // Slightly slower for better comprehension
  utterance.pitch = 1.0;
  utterance.volume = 1.0;

  // Try to use a more natural voice if available
  const voices = speechSynthesis.getVoices();
  const preferredVoices = voices.filter(
    (voice) =>
      voice.lang.startsWith("en") &&
      (voice.name.includes("Natural") ||
        voice.name.includes("Enhanced") ||
        voice.name.includes("Premium") ||
        voice.name.includes("Google") ||
        voice.name.includes("Microsoft"))
  );

  if (preferredVoices.length > 0) {
    utterance.voice = preferredVoices[0];
  } else if (voices.length > 0) {
    // Fallback to first English voice
    const englishVoice = voices.find((voice) => voice.lang.startsWith("en"));
    if (englishVoice) {
      utterance.voice = englishVoice;
    }
  }

  // Update button state
  btn.classList.add("listening");
  btn.innerHTML = "⏹ Stop";
  currentSpeech = utterance;

  // Handle speech events
  utterance.onend = () => {
    btn.classList.remove("listening");
    btn.innerHTML = "🔊 Listen";
    currentSpeech = null;
  };

  utterance.onerror = (event) => {
    console.error("Speech synthesis error:", event.error);
    btn.classList.remove("listening");
    btn.innerHTML = "🔊 Listen";
    currentSpeech = null;

    // Show user-friendly error message
    if (event.error === "network") {
      alert(
        "Network error occurred while trying to speak. Please check your connection and try again."
      );
    } else {
      alert(
        "An error occurred while trying to speak the text. Please try again."
      );
    }
  };

  utterance.onpause = () => {
    btn.classList.remove("listening");
    btn.innerHTML = "🔊 Listen";
  };

  // Start speaking
  speechSynthesis.speak(utterance);
}

// Load voices when they become available (some browsers load voices asynchronously)
if (ttsSupported) {
  speechSynthesis.onvoiceschanged = () => {
    // Voices are now loaded and available
  };
}

// Analyze sentiment function
function analyzeSentiment(summaryText) {
  // Get current theme and add to form
  const currentTheme = localStorage.getItem("news-analyzer-theme") || "light";

  // Redirect to sentiment analysis page
  const form = document.createElement("form");
  form.method = "POST";
  form.action = `/sentiment_analysis?theme=${currentTheme}`;
  form.style.display = "none";

  const textInput = document.createElement("input");
  textInput.type = "hidden";
  textInput.name = "summary_text";
  textInput.value = summaryText;

  form.appendChild(textInput);
  document.body.appendChild(form);
  form.submit();
}

// Event listeners
inputText.addEventListener("input", updateUI);

resetBtn.addEventListener("click", () => {
  inputText.value = "";
  updateUI();
  showEmptyState();
});

summarizeBtn.addEventListener("click", async () => {
  if (!inputText.value.trim()) return;

  showLoading();

  try {
    const summary = await summarizeText(inputText.value, selectedModel);
    if (summary && summary.trim()) {
      showSummary(summary);
    } else {
      showError("Unable to generate summary");
    }
  } catch (error) {
    showError(error.message || "Something went wrong");
  }
});

// Keyboard shortcut (Ctrl/Cmd + Enter to summarize)
inputText.addEventListener("keydown", function (event) {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    if (!summarizeBtn.disabled) {
      summarizeBtn.click();
    }
  }
});

// Stop speech when page is about to unload
window.addEventListener("beforeunload", () => {
  if (speechSynthesis.speaking) {
    speechSynthesis.cancel();
  }
});

// Stop speech when page loses focus (optional - for better UX)
document.addEventListener("visibilitychange", () => {
  if (document.hidden && speechSynthesis.speaking) {
    speechSynthesis.pause();
  }
});

// Add shake animation for errors
const shakeKeyframes = `
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                25% { transform: translateX(-5px); }
                75% { transform: translateX(5px); }
            }
        `;

if (!document.querySelector("#shake-animation")) {
  const style = document.createElement("style");
  style.id = "shake-animation";
  style.textContent = shakeKeyframes;
  document.head.appendChild(style);
}

// Initialize UI
updateUI();
showEmptyState();
