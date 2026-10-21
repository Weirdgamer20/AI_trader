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
        <span class="hud-val text-cyan" id="hud-signal">DATA UNAVAILABLE</span>
      </div>
      <div class="hud-metric-row">
        <span class="hud-label">Expected Net Ret:</span>
        <span class="hud-val" id="hud-net-return">--</span>
      </div>
      <div class="hud-metric-row">
        <span class="hud-label">Uncertainty Band:</span>
        <span class="hud-val" id="hud-uncertainty">--</span>
      </div>
      <div class="hud-metric-row">
        <span class="hud-label">Planned Risk/Trade:</span>
        <span class="hud-val" id="hud-risk">Max 2.0% (Strict Cap)</span>
      </div>
      <div class="hud-metric-row">
        <span class="hud-label">Loss Ceiling:</span>
        <span class="hud-val text-cyan" id="hud-ceiling">₹250 Hard Limit</span>
      </div>
      <div class="hud-tags" id="hud-tags">
        <span class="hud-tag">AWAITING_LOCAL_API</span>
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
        if (data.tags && Array.isArray(data.tags)) {
          const tagsContainer = document.getElementById('hud-tags');
          tagsContainer.textContent = '';
          data.tags.forEach(t => {
            const span = document.createElement('span');
            span.className = 'hud-tag';
            span.textContent = String(t);
            tagsContainer.appendChild(span);
          });
        }
      } else {
        document.getElementById('hud-signal').textContent = 'LOCAL API ERROR';
      }
    } catch (e) {
      document.getElementById('hud-signal').textContent = 'DATA UNAVAILABLE (OFFLINE)';
      document.getElementById('hud-net-return').textContent = '--';
      document.getElementById('hud-uncertainty').textContent = '--';
    }
  }

  updateOverlayData();
  setInterval(updateOverlayData, 10000);
})();
