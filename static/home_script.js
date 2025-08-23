// Theme Toggle Functionality
const themeToggle = document.getElementById("themeToggle");
const body = document.body;

// Always default to 'light' theme on page load
body.setAttribute("data-theme", "light");
themeToggle.checked = true;
localStorage.setItem("theme", "light");

themeToggle.addEventListener("change", function () {
  if (this.checked) {
    body.setAttribute("data-theme", "light");
    localStorage.setItem("theme", "light");
  } else {
    body.removeAttribute("data-theme");
    localStorage.setItem("theme", "dark");
  }
});

// Alternative: Use NewsAPI.org (free tier available)
// Sign up at https://newsapi.org/ and replace 'YOUR_NEWSAPI_KEY' with your actual key
// const NEWS_API_KEY = 'YOUR_NEWSAPI_KEY';
// const NEWSAPI_URL = `https://newsapi.org/v2/top-headlines?country=us&pageSize=5&apiKey=${NEWS_API_KEY}`;
// const API_URL = `https://api.allorigins.win/raw?url=${encodeURIComponent(NEWSAPI_URL)}`;

// Using GNews with CORS proxy
const API_KEY = "133bb93beaeb9f5cfc91ea8efee7b0c4";
const GNEWS_URL = `https://gnews.io/api/v4/top-headlines?token=${API_KEY}&lang=en&max=5`;
const API_URL = `https://api.allorigins.win/raw?url=${encodeURIComponent(
  GNEWS_URL
)}`;

let newsCache = null;
let lastFetchTime = 0;
const CACHE_DURATION = 10 * 60 * 1000; // 10 minutes

async function fetchNews() {
  const now = Date.now();

  // Use cache if available and not expired
  if (newsCache && now - lastFetchTime < CACHE_DURATION) {
    return newsCache;
  }

  try {
    const response = await fetch(API_URL);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();

    // Cache the results
    newsCache = data;
    lastFetchTime = now;

    return data;
  } catch (error) {
    console.error("Error fetching news:", error);
    throw error;
  }
}

function formatTimeAgo(dateString) {
  const now = new Date();
  const publishTime = new Date(dateString);
  const diffInMinutes = Math.floor((now - publishTime) / (1000 * 60));

  if (diffInMinutes < 1) return "Just now";
  if (diffInMinutes < 60) return `${diffInMinutes}m ago`;

  const diffInHours = Math.floor(diffInMinutes / 60);
  if (diffInHours < 24) return `${diffInHours}h ago`;

  const diffInDays = Math.floor(diffInHours / 24);
  if (diffInDays < 7) return `${diffInDays}d ago`;

  return publishTime.toLocaleDateString();
}

function displayNews(articles) {
  const container = document.getElementById("newsContainer");

  if (!articles || articles.length === 0) {
    container.innerHTML = `
                    <div class="error-message">
                        <div class="error-icon">📰</div>
                        <h3>No news articles available</h3>
                        <p>Unable to fetch news articles at the moment. Please try again later.</p>
                    </div>
                `;
    return;
  }

  const newsHTML = articles
    .map((article) => {
      const timeAgo = formatTimeAgo(article.publishedAt);
      const source = article.source?.name || "Unknown Source";

      return `
                    <div class="news-item">
                        <h3><a href="${article.url}" target="_blank" rel="noopener noreferrer">${article.title}</a></h3>
                        <div class="news-meta">
                            <span class="news-source">${source}</span>
                            <span class="news-time">🕐 ${timeAgo}</span>
                        </div>
                    </div>
                `;
    })
    .join("");

  container.innerHTML = `<div class="news-grid">${newsHTML}</div>`;
}

function displayError() {
  const container = document.getElementById("newsContainer");
  container.innerHTML = `
                <div class="error-message">
                    <div class="error-icon">⚠️</div>
                    <h3>Unable to fetch news</h3>
                    <p>Unable to fetch news at the moment. Please try again later.</p>
                </div>
            `;
}

function showLoading() {
  const container = document.getElementById("newsContainer");
  container.innerHTML = `
                <div class="loading-spinner">
                    <div class="spinner"></div>
                </div>
            `;
}

async function loadNews() {
  const refreshBtn = document.getElementById("refreshNews");
  const refreshIcon = refreshBtn.querySelector(".refresh-icon");

  try {
    refreshBtn.classList.add("loading");
    refreshBtn.disabled = true;
    showLoading();

    const data = await fetchNews();
    displayNews(data.articles);
  } catch (error) {
    console.error("Error loading news:", error);
    displayError();
  } finally {
    refreshBtn.classList.remove("loading");
    refreshBtn.disabled = false;
  }
}

// Event Listeners
document.getElementById("refreshNews").addEventListener("click", () => {
  // Clear cache to force fresh fetch
  newsCache = null;
  lastFetchTime = 0;
  loadNews();
});

// Load news on page load
document.addEventListener("DOMContentLoaded", loadNews);

// Auto-refresh news every 10 minutes
setInterval(() => {
  loadNews();
}, 10 * 60 * 1000);
