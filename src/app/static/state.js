// Cross-module mutable app state, cached DOM refs, and localStorage helpers.
//
// `S` holds the handful of scalars that more than one module reads/writes
// (the view, model). Everything else stays local to its owning module. `el`
// is populated once by cacheDom() after the document is parsed — modules
// must not touch the DOM at import time.
//
// The inference device is chosen once at server startup (see src/app/app.py
// --device) and is not app state here — there is no in-UI device switch.

export const STORE = 'airfoilsurrogate.v1.';
export function save(k, v){ try{ localStorage.setItem(STORE + k, v); }catch(e){} }
export function load(k){ try{ return localStorage.getItem(STORE + k); }catch(e){ return null; } }

export const S = {
  currentView: 'streamlines',
  currentModel: 'gno',
};

export const el = {};
export function cacheDom(){
  const id = (x) => document.getElementById(x);
  el.app = id('app');
  el.designHolder = id('designHolder');
  el.designCanvas = id('designCanvas');
  el.designCB = id('designCB');
  el.cbTicks = id('cbTicks');
  el.cbTitle = id('cbTitle');
  el.toast = id('toast');
  el.aoa = id('aoa');
  el.aoaVal = id('aoaVal');
  el.nacaInput = id('nacaInput');
  el.reInput = id('reInput');
  el.genBtn = id('genBtn');
  el.vizDD = id('vizDD');
  el.exportStepBtn = id('exportStepBtn');
}
