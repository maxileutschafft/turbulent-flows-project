// Entry point: cache the DOM, wire every control, restore persisted state, and
// load the initial geometry. Module scripts are deferred, so the document is
// already parsed when this runs.

import { S, el, save, load, cacheDom } from './state.js';
import * as core from './core.js';

cacheDom();
core.initCore();          // snapshot the field-view menu; sync designAlpha

/* ---------- global chrome ---------- */
document.addEventListener('click', function(){ core.closeMenus(); });
window.addEventListener('resize', core.fitAll);

/* ---------- view + model dropdowns ---------- */
core.wireDropdown('vizDD', function(v){ core.showView(v); save('viz', v); });
core.wireDropdown('modelDD', function(v){ S.currentModel = v.toLowerCase(); save('model', v); core.clearPrediction(); });

/* ---------- inputs ---------- */
el.aoa.addEventListener('input', core.updAoa); core.updAoa();

let nacaT;
el.nacaInput.addEventListener('input', function(){
  core.clearPrediction();
  clearTimeout(nacaT);
  nacaT = setTimeout(function(){ const code = el.nacaInput.value.trim(); if(/^\d{4}$/.test(code)) core.fetchGeometry(code); }, 250);
});
el.reInput.addEventListener('input', function(){ core.clearPrediction(); });
el.genBtn.addEventListener('click', core.generatePrediction);

/* ---------- zoom ---------- */
document.querySelectorAll('.zoom button').forEach(function(b){
  b.addEventListener('click', function(){ core.stepZoom(parseInt(b.dataset.z, 10)); });
});

/* ---------- device dropdown ---------- */
(function fetchDevices(){
  fetch('/api/devices')
    .then(function(r){ if(!r.ok) throw new Error('http ' + r.status); return r.json(); })
    .then(function(d){
      const menu = document.querySelector('#deviceDD .dd-menu');
      const label = document.querySelector('#deviceDD .dd-label');
      const saved = load('device');
      const preferred = (saved && d.available.indexOf(saved)!==-1) ? saved : d.default;
      d.available.forEach(function(dev){
        const lbl = (d.labels&&d.labels[dev]) ? d.labels[dev] : dev.charAt(0).toUpperCase()+dev.slice(1);
        const item = document.createElement('div');
        item.className = 'dd-item' + (dev===preferred?' sel':'');
        item.dataset.value = dev;
        item.textContent = lbl;
        menu.appendChild(item);
      });
      const prefLabel = document.querySelector('#deviceDD .dd-item[data-value="' + preferred + '"]');
      if(prefLabel&&label) label.textContent = prefLabel.textContent;
      S.currentDevice = preferred;
      core.wireDropdown('deviceDD', function(v){ S.currentDevice = v; save('device', v); });
    })
    .catch(function(){}); // silently ignore; currentDevice stays '' → backend default
})();

/* ---------- restore persisted state ---------- */
(function restore(){
  const model = load('model'); if(model){ const mi = document.querySelector('#modelDD .dd-item[data-value="' + model + '"]'); if(mi) mi.click(); core.closeMenus(); }
  const v = load('viz'); if(v){ const vi = document.querySelector('#vizDD .dd-item[data-value="' + v + '"]'); if(vi){ vi.click(); core.closeMenus(); } }
})();

/* ---------- initial geometry load ---------- */
core.fetchGeometry(el.nacaInput.value.trim());
requestAnimationFrame(function(){ core.fitAll(); setTimeout(core.fitAll, 120); });
