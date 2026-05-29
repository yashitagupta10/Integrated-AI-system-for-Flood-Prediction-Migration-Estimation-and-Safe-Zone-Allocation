/**
 * frontend_realtime_patch.js
 * ==========================
 * Add this to your static/js/main.js (or inline in your template).
 *
 * What it does:
 *   1. When user selects a district → auto-fetches rainfall & affected area
 *   2. Updates the slider + input field automatically
 *   3. Shows a live data badge with source and timestamp
 *   4. Calls /analyze_hybrid instead of /analyze for the hybrid model
 */

// ── Auto-fetch rainfall when district is selected ────────────────────────────
const districtSelect = document.getElementById('district-select');
// Adjust selector ↑ to match your actual dropdown element ID

const REALTIME_BADGE_HTML = `
  <span id="realtime-badge" style="
    background: #0ea5e9; color: white; font-size: 11px; 
    padding: 2px 8px; border-radius: 12px; margin-left: 8px;
    animation: pulse 1.5s infinite;
  ">⚡ LIVE</span>`;

const LOADING_BADGE_HTML = `
  <span id="realtime-badge" style="
    background: #6b7280; color: white; font-size: 11px;
    padding: 2px 8px; border-radius: 12px; margin-left: 8px;
  ">⏳ Fetching...</span>`;

async function fetchRealtimeWeather(district) {
  if (!district || district === '') return;

  // Show loading state
  showRainfallBadge(LOADING_BADGE_HTML);
  setInputsDisabled(true);

  try {
    const resp = await fetch(`/get_realtime_weather/${encodeURIComponent(district)}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    if (!data.success) throw new Error(data.error || 'Unknown error');

    // ── Update rainfall slider ──────────────────────────────────────────
    const rainfallSlider = document.querySelector('input[type="range"]');
    if (rainfallSlider) {
      rainfallSlider.value = Math.min(data.rainfall_mm, 500);
      rainfallSlider.dispatchEvent(new Event('input'));
    }

    // ── Update rainfall text input ──────────────────────────────────────
    const rainfallInput = document.querySelector('input[placeholder*="type value"]');
    if (rainfallInput) {
      rainfallInput.value = data.rainfall_mm.toFixed(1);
      rainfallInput.dispatchEvent(new Event('change'));
    }

    // ── Update affected area input ──────────────────────────────────────
    const areaInput = document.querySelector('input[placeholder*="25"]');
    if (areaInput) {
      areaInput.value = data.affected_area_km2.toFixed(1);
      areaInput.dispatchEvent(new Event('change'));
    }

    // ── Show live badge with timestamp ──────────────────────────────────
    const ts = new Date(data.timestamp).toLocaleTimeString('en-IN', {
      hour: '2-digit', minute: '2-digit'
    });
    showRainfallBadge(`
      <span id="realtime-badge" title="Data from Open-Meteo at ${data.timestamp}" style="
        background: linear-gradient(90deg, #0ea5e9, #06b6d4);
        color: white; font-size: 11px; padding: 2px 8px;
        border-radius: 12px; margin-left: 8px; cursor: help;
      ">⚡ LIVE · ${ts} · ${data.rainfall_mm.toFixed(1)}mm</span>
    `);

    // ── Show 7-day forecast mini chart (if forecast container exists) ───
    renderForecastBadge(data.forecast_7day);

    console.log(`[FloodSense] Loaded real-time data for ${district}:`, data);

  } catch (err) {
    console.warn('[FloodSense] Real-time fetch failed, manual entry enabled:', err);
    showRainfallBadge(`
      <span id="realtime-badge" style="
        background: #dc2626; color: white; font-size: 11px;
        padding: 2px 8px; border-radius: 12px; margin-left: 8px;
      ">⚠ Manual mode</span>
    `);
  } finally {
    setInputsDisabled(false);
  }
}

function showRainfallBadge(html) {
  const existing = document.getElementById('realtime-badge');
  if (existing) existing.remove();

  // Insert badge after the RAINFALL label
  const rainfallLabel = document.querySelector('label, .label, [class*="rainfall"]');
  if (rainfallLabel) {
    rainfallLabel.insertAdjacentHTML('beforeend', html);
  }
}

function setInputsDisabled(disabled) {
  const slider = document.querySelector('input[type="range"]');
  const textInput = document.querySelector('input[placeholder*="type value"]');
  const areaInput = document.querySelector('input[placeholder*="25"]');
  [slider, textInput, areaInput].forEach(el => {
    if (el) el.disabled = disabled;
  });
}

function renderForecastBadge(forecastArr) {
  // Renders a tiny inline 7-day bar chart
  if (!forecastArr || forecastArr.length === 0) return;

  let existing = document.getElementById('forecast-mini');
  if (!existing) {
    existing = document.createElement('div');
    existing.id = 'forecast-mini';
    existing.style.cssText = `
      display: flex; gap: 3px; align-items: flex-end; height: 30px;
      margin-top: 6px; padding: 4px 8px; background: rgba(255,255,255,0.05);
      border-radius: 6px;
    `;
    const areaInput = document.querySelector('input[placeholder*="25"]');
    if (areaInput && areaInput.parentElement) {
      areaInput.parentElement.appendChild(existing);
    }
  }

  const max = Math.max(...forecastArr, 1);
  const days = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];
  existing.innerHTML = forecastArr.map((v, i) => {
    const h = Math.max(4, Math.round((v / max) * 26));
    const color = v > 100 ? '#ef4444' : v > 50 ? '#f59e0b' : '#22c55e';
    return `<div title="Day ${i + 1}: ${v}mm" style="
      width: 12px; height: ${h}px; background: ${color};
      border-radius: 2px 2px 0 0; flex-shrink: 0;
    "></div>`;
  }).join('') + `<span style="color:#9ca3af;font-size:9px;margin-left:4px;align-self:center;">7d</span>`;
}

// ── Wire up district dropdown ────────────────────────────────────────────────
if (districtSelect) {
  districtSelect.addEventListener('change', (e) => {
    fetchRealtimeWeather(e.target.value);
  });

  // Also handle custom dropdowns (React/custom select libraries)
  // Poll for value change as fallback
  let lastDistrict = '';
  setInterval(() => {
    const current = districtSelect.value;
    if (current && current !== lastDistrict) {
      lastDistrict = current;
      fetchRealtimeWeather(current);
    }
  }, 300);
}

// ── Replace analyze button handler with hybrid model ────────────────────────
const analyzeBtn = document.getElementById('analyze-btn');
// Adjust ↑ to match your actual Analyze & Predict button ID

if (analyzeBtn) {
  analyzeBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    e.stopPropagation(); // prevent original handler if needed

    const district = districtSelect ? districtSelect.value : '';
    const rainfallInput = document.querySelector('input[placeholder*="type value"]');
    const areaInput = document.querySelector('input[placeholder*="25"]');

    if (!district) {
      alert('Please select a district first.');
      return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '⚡ Analyzing with AI...';

    try {
      const resp = await fetch('/analyze_hybrid', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          district,
          rainfall: parseFloat(rainfallInput?.value || 0),
          affected_area: parseFloat(areaInput?.value || 50),
          use_realtime: true,  // always pull latest data
        }),
      });

      const result = await resp.json();
      console.log('[FloodSense] Hybrid prediction result:', result);

      // ── Render result (calls your existing render function) ────────────
      if (typeof renderAnalysisResult === 'function') {
        renderAnalysisResult(result);
      } else {
        // Fallback: dispatch custom event for existing handlers to catch
        document.dispatchEvent(new CustomEvent('floodsense:result', { detail: result }));
      }

    } catch (err) {
      console.error('[FloodSense] Analyze failed:', err);
      alert('Analysis failed. Check console for details.');
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = '⚡ ANALYZE & PREDICT';
    }
  });
}

// ── CSS animation for live badge ─────────────────────────────────────────────
const style = document.createElement('style');
style.textContent = `
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }
`;
document.head.appendChild(style);

console.log('[FloodSense] Real-time weather module loaded.');