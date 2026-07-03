// Screen-projection constants + airfoil silhouette drawing.
// Pure: no app state, no DOM. Renders the live /api/geometry coordinates into
// the 480x260 canvas frame (the dashed chord reference + the airfoil outline).

export const ALPHA = 8, S = 200, OX = 130, CY = 130, PIV = 0.25;

export function Wrot(x, y, a){
  const A = Math.PI / 180 * a, ca = Math.cos(A), sa = Math.sin(A);
  const dx = x - PIV, dy = y;
  const xr = PIV + dx * ca + dy * sa, yr = -dx * sa + dy * ca;
  return [OX + xr * S, CY - yr * S];
}

export function geometrySVG(c, a){
  const pts = [];
  for (let i = 0; i < c.xu.length; i++) pts.push(Wrot(c.xu[i], c.yu[i], a));      // upper LE->TE
  for (let i = c.xl.length - 1; i >= 0; i--) pts.push(Wrot(c.xl[i], c.yl[i], a)); // lower TE->LE
  const f = (p) => p[0].toFixed(1) + ' ' + p[1].toFixed(1);
  const afD = 'M ' + pts.map(f).join(' L ') + ' Z';
  const pv = Wrot(PIV, 0, a);                                                     // pivot fixed -> reference horizontal
  const refD = 'M ' + (pv[0] - 95).toFixed(1) + ' ' + pv[1].toFixed(1) +
               ' L ' + (pv[0] + 150).toFixed(1) + ' ' + pv[1].toFixed(1);
  const inner = '<path d="' + refD + '" fill="none" stroke="#9aa0a8" stroke-width="1" stroke-dasharray="3 4"/>'
    + '<path d="' + afD + '" fill="#ffffff" stroke="#1d2025" stroke-width="1.5" stroke-linejoin="round"/>';
  return '<svg class="viz-svg active" viewBox="0 0 480 260" preserveAspectRatio="xMidYMid meet">' + inner + '</svg>';
}
