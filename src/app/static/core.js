// Canvas rendering, floating chrome (toast / dropdowns / zoom / colorbar),
// and the predicted-field path for the Design view.

import { S, el, save } from './state.js';
import { ALPHA, geometrySVG } from './geometry.js';

// ---- canvas / prediction state (module-local) ----
let designCoords = null, designAlpha = ALPHA;
let designFieldSvg = null, streamlineAoa = null, designColorbar = null;
let predictionActive = false;
let predNaca = null, predRe = null, predAoa = null, predModel = null;
let fieldCache = {};
let toastT;

export function initCore(){
  designAlpha = parseFloat(el.aoa.value);
}

// ---- render ----
export function renderDesignGeom(){
  if (!designCoords) return;
  const fieldDiv = designFieldSvg ? '<div class="viz-svg active" style="position:absolute;inset:0;">' + designFieldSvg + '</div>' : '';
  el.designHolder.innerHTML = fieldDiv + geometrySVG(designCoords, designAlpha);
  applyZoom();
}

export function fetchGeometry(code){
  return fetch('/api/geometry?naca=' + encodeURIComponent(code))
    .then(function(r){ if(!r.ok) throw new Error('http ' + r.status); return r.json(); })
    .then(function(d){ designCoords = d; renderDesignGeom(); })
    .catch(function(){ toast('Enter a valid 4-digit NACA code'); });
}

// ---- holder fit ----
export function fit(holder){
  if(!holder) return; const area = holder.parentElement;
  const r = area.getBoundingClientRect(); const cs = getComputedStyle(area);
  const pad = parseFloat(cs.paddingLeft) || 0;
  const aw = r.width - 2*pad, ah = r.height - 2*pad; if(aw<=0||ah<=0) return;
  const ar = 480/260; let w = aw, h = aw/ar; if(h>ah){ h = ah; w = ah*ar; }
  holder.style.width = Math.round(w) + 'px'; holder.style.height = Math.round(h) + 'px';
}
export function fitAll(){ fit(el.designHolder); }

// ---- toast ----
export function toast(msg){
  el.toast.textContent = msg; el.toast.classList.add('show');
  clearTimeout(toastT); toastT = setTimeout(function(){ el.toast.classList.remove('show'); }, 2200);
}

// ---- generic dropdown ----
export function closeMenus(except){
  document.querySelectorAll('.dropdown.open').forEach(function(d){ if(d!==except) d.classList.remove('open'); });
}
export function wireDropdown(id, onSelect){
  const dd = document.getElementById(id); if(!dd) return;
  const btn = dd.querySelector('.dd-btn'), label = dd.querySelector('.dd-label');
  btn.addEventListener('click', function(e){ e.stopPropagation(); const willOpen = !dd.classList.contains('open'); closeMenus(dd); dd.classList.toggle('open', willOpen); });
  dd.querySelectorAll('.dd-item').forEach(function(it){
    it.addEventListener('click', function(e){ e.stopPropagation();
      dd.querySelectorAll('.dd-item').forEach(function(x){ x.classList.remove('sel'); });
      it.classList.add('sel');
      if(label) label.textContent = it.dataset.label || it.textContent.trim();
      dd.classList.remove('open');
      if(onSelect) onSelect(it.dataset.value, it);
    });
  });
}

export function applyColorbar(cb){
  if(!cb){ el.designCB.classList.remove('show'); return; }
  if(cb.label){ el.cbTitle.textContent = cb.label; el.cbTitle.style.display=''; }
  else { el.cbTitle.style.display='none'; }
  el.cbTicks.innerHTML = cb.ticks.map(function(t){ return '<span>' + t + '</span>'; }).join('');
  const bar = el.designCB.querySelector('.cb-bar');
  if(bar){ bar.style.background = cb.gradient ? 'linear-gradient(to top,' + cb.gradient.join(',') + ')' : ''; }
  el.designCB.classList.add('show');
}

export function clearPrediction(){
  predictionActive = false; designFieldSvg = null; designColorbar = null; streamlineAoa = null;
  fieldCache = {}; applyColorbar(null); renderDesignGeom();
}

export function showView(view){
  S.currentView = view;
  if(!predictionActive){
    designFieldSvg = null; designColorbar = null; applyColorbar(null); renderDesignGeom(); return;
  }
  const key = predNaca + '|' + predRe + '|' + predAoa + '|' + view + '|' + predModel;
  if(fieldCache[key]){
    designFieldSvg = fieldCache[key].field_svg; designColorbar = fieldCache[key].colorbar;
    applyColorbar(designColorbar); renderDesignGeom(); return;
  }
  el.designCanvas.classList.add('predicting');
  fetch('/api/predict?naca=' + encodeURIComponent(predNaca) + '&reynolds=' + predRe + '&aoa=' + predAoa + '&view=' + encodeURIComponent(view) + '&model=' + predModel + '&device=' + S.currentDevice)
    .then(function(r){ if(!r.ok) return r.json().then(function(e){ throw new Error(e.detail||('HTTP '+r.status)); }); return r.json(); })
    .then(function(data){
      fieldCache[key] = { field_svg: data.field_svg, colorbar: data.colorbar };
      designFieldSvg = data.field_svg; designColorbar = data.colorbar;
      applyColorbar(designColorbar); renderDesignGeom();
    })
    .catch(function(err){ toast('View failed: ' + (err&&err.message||err)); })
    .finally(function(){ el.designCanvas.classList.remove('predicting'); });
}

// ---- Reynolds parsing + case inputs ----
export function parseReynolds(str){
  // map unicode superscript digits to ASCII
  const s = str.replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹]/g, function(c){
    return '0123456789'.charAt('⁰¹²³⁴⁵⁶⁷⁸⁹'.indexOf(c));
  }).trim();
  // match "mantissa (× or x) 10 ^? exponent"
  const m = s.match(/^([0-9]*\.?[0-9]+)\s*[×x\*]\s*10\s*\^?\s*([0-9]+)$/i);
  if(m) return parseFloat(m[1]) * Math.pow(10, parseInt(m[2], 10));
  return parseFloat(s.replace(/\s/g, ''));
}
export function caseInputs(){
  const naca = el.nacaInput.value.trim();
  if(!/^\d{4}$/.test(naca)) return null;
  const reynolds = parseReynolds(el.reInput.value);
  if(isNaN(reynolds)||reynolds<=0) return null;
  return { naca: naca, reynolds: reynolds, aoa: parseFloat(el.aoa.value) };
}

// ---- AoA slider ----
export function updAoa(){
  const v = parseFloat(el.aoa.value); el.aoaVal.textContent = v.toFixed(1) + '°';
  const pct = (v-(-10))/30*100; el.aoa.style.setProperty('--pct', pct + '%'); designAlpha = v;
  if(streamlineAoa!==null && v!==streamlineAoa){ clearPrediction(); } else { renderDesignGeom(); }
}

// ---- generate prediction ----
export function generatePrediction(){
  if(el.genBtn.disabled) return;
  const naca = el.nacaInput.value.trim();
  if(!/^\d{4}$/.test(naca)){ toast('Enter a valid 4-digit NACA code'); return; }
  const reynolds = parseReynolds(el.reInput.value);
  if(isNaN(reynolds)||reynolds<=0){ toast('Enter a valid Reynolds number'); return; }
  const aoaVal2 = parseFloat(el.aoa.value);
  el.genBtn.disabled = true;
  const origHtml = el.genBtn.innerHTML; el.genBtn.innerHTML = '<span class="mini-spin"></span>Predicting…';
  el.designCanvas.classList.add('predicting');
  fetch('/api/predict?naca=' + encodeURIComponent(naca) + '&reynolds=' + reynolds + '&aoa=' + aoaVal2 + '&view=' + encodeURIComponent(S.currentView) + '&model=' + S.currentModel + '&device=' + S.currentDevice)
    .then(function(r){ if(!r.ok) return r.json().then(function(e){ throw new Error(e.detail||('HTTP '+r.status)); }); return r.json(); })
    .then(function(data){
      // sync AoA slider/label to predicted AoA WITHOUT clearing streamlines (avoid updAoa clearing logic)
      const gotAoa = data.aoa;
      el.aoa.value = gotAoa;
      el.aoaVal.textContent = gotAoa.toFixed(1) + '°';
      const pct = (gotAoa-(-10))/30*100; el.aoa.style.setProperty('--pct', pct + '%');
      designAlpha = gotAoa;
      // store prediction identity + cache result
      predictionActive = true; predNaca = naca; predRe = reynolds; predAoa = gotAoa; predModel = S.currentModel;
      const key = naca + '|' + reynolds + '|' + gotAoa + '|' + S.currentView + '|' + S.currentModel;
      fieldCache[key] = { field_svg: data.field_svg, colorbar: data.colorbar };
      // assign field svg AFTER syncing slider so updAoa guard doesn't wipe it
      designFieldSvg = data.field_svg; designColorbar = data.colorbar; streamlineAoa = gotAoa;
      applyColorbar(data.colorbar);
      renderDesignGeom();
      toast('Prediction · ' + data.compute_ms + ' ms');
    })
    .catch(function(err){ toast('Prediction failed: ' + (err&&err.message||err)); })
    .finally(function(){ el.designCanvas.classList.remove('predicting'); el.genBtn.innerHTML = origHtml; el.genBtn.disabled = false; });
}

// ---- zoom ----
const zlevels = [60,75,90,100,115,130,150]; let zi = 3;
export function applyZoom(){
  const z = zlevels[zi]/100;
  document.querySelectorAll('.viz-svg').forEach(function(l){ l.style.transform = 'scale(' + z + ')'; l.style.transformOrigin = 'center'; });
  document.querySelectorAll('.zval').forEach(function(e){ e.textContent = zlevels[zi] + '%'; });
}
export function stepZoom(z){ if(z===0){ zi = 3; } else { zi = Math.max(0, Math.min(zlevels.length-1, zi+z)); } applyZoom(); }
