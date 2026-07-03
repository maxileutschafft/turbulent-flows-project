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
el.exportStepBtn.addEventListener('click', core.exportStep);

/* ---------- zoom ---------- */
document.querySelectorAll('.zoom button').forEach(function(b){
  b.addEventListener('click', function(){ core.stepZoom(parseInt(b.dataset.z, 10)); });
});

/* ---------- restore persisted state ---------- */
(function restore(){
  const model = load('model'); if(model){ const mi = document.querySelector('#modelDD .dd-item[data-value="' + model + '"]'); if(mi) mi.click(); core.closeMenus(); }
  const v = load('viz'); if(v){ const vi = document.querySelector('#vizDD .dd-item[data-value="' + v + '"]'); if(vi){ vi.click(); core.closeMenus(); } }
})();

/* ---------- initial geometry load ---------- */
core.fetchGeometry(el.nacaInput.value.trim());
requestAnimationFrame(function(){ core.fitAll(); setTimeout(core.fitAll, 120); });
