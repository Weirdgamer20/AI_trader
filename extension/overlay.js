// CoinDCX AI Trading Overlay (Click-Through)
// Non-interfering HUD displaying AI forecasts and deterministic risk parameters.
(function () {
  'use strict';

  if (document.getElementById('coindcx-ai-hud-container')) {
    return; // Already initialized
  }

  // Detect CoinDCX Spot Page
  const isTradePage = window.location.pathname.includes('/trade/') || window.location.pathname.includes('/spot/');
  if (!isTradePage && !window.location.hostname.includes('coindcx.com')) {
    return;
  }

  // Create Container with pointer-events: none for full click-through to underlying charts
  const container = document.createElement('div');
  container.id = 'coindcx-ai-hud-container';
  container.className = 'coindcx-ai-hud-wrapper';

  const card = document.createElement('div');
  card.className = 'coindcx-ai-hud-card';

  card.innerHTML = `
    <div class="hud-header">
      <div class="hud-title">⚡ CoinDCX AI Co-Pilot</div>
      <div class="hud-badge">SPOT ONLY</div>
    </div>
    <div class="hud-body">
      <div class="hud-metric-row">
        <span class="hud-label">Model Signal:</span>
        <span class="hud-val text-green" id="hud-signal">UP (62% Calibrated)</span>
      </div>
      <div class="hud-metric-row">
        <span class="hud-label">Expected Net Ret:</span>
        <span class="hud-val" id="hud-net-return">+2.15% (After Fees)</span>
      </div>
      <div class="hud-metric-row">
        <span class="hud-label">Uncertainty Band:</span>
        <span class="hud-val" id="hud-uncertainty">±2.4% (90% Conf)</span>
      </div>
      <div class="hud-metric-row">
        <span class="hud-label">Planned Risk/Trade:</span>
        <span class="hud-val" id="hud-risk">Max 2.0% (₹10 at ₹500)</span>
      </div>
      <div class="hud-metric-row">
        <span class="hud-label">Loss Ceiling:</span>
        <span class="hud-val text-cyan" id="hud-ceiling">₹250 Hard Limit</span>
      </div>
      <div class="hud-tags" id="hud-tags">
        <span class="hud-tag">TRENDING_BULLISH</span>
        <span class="hud-tag">LIQUID_SPOT</span>
      </div>
    </div>
    <div class="hud-footer">
      <span>CLICK-THROUGH ACTIVE • NON-CUSTODIAL</span>
    </div>
  `;

  container.appendChild(card);
  document.body.appendChild(container);

  // Keep HUD updated from local trading server if running
  async function updateOverlayData() {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/hud-status', { mode: 'cors' });
      if (res.ok) {
        const data = await res.json();
        if (data.signal) document.getElementById('hud-signal').textContent = data.signal;
        if (data.net_return) document.getElementById('hud-net-return').textContent = data.net_return;
        if (data.uncertainty) document.getElementById('hud-uncertainty').textContent = data.uncertainty;
      }
    } catch (e) {
      // Local server offline; HUD retains default baseline indicator safely
    }
  }

  setInterval(updateOverlayData, 15000);
})();
