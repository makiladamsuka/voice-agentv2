const fs = require('fs');

let css = fs.readFileSync('frontend/styles/globals.css', 'utf8');

const lightVars = `
  --surface-bright: #fdf8fd;
  --tertiary-fixed: #f7e0ad;
  --surface-tint: #4e6076;
  --on-primary-fixed: #081d30;
  --error: #ba1a1a;
  --on-tertiary: #ffffff;
  --on-primary: #ffffff;
  --on-secondary-container: #785a65;
  --on-primary-fixed-variant: #36485e;
  --outline: #74777d;
  --inverse-surface: #313034;
  --on-tertiary-fixed: #241a00;
  --on-surface: #1c1b1f;
  --on-error: #ffffff;
  --on-background: #1c1b1f;
  --tertiary-container: #f7e0ad;
  --tertiary: #6d5d34;
  --on-surface-variant: #44474c;
  --on-secondary-fixed-variant: #5a3f49;
  --secondary-fixed: #ffd9e4;
  --surface-variant: #e5e1e7;
  --surface-container-low: #f7f2f8;
  --primary: #4e6076;
  --on-tertiary-fixed-variant: #54451f;
  --primary-fixed: #d1e4ff;
  --on-primary-container: #54667d;
  --on-tertiary-container: #73623a;
  --surface-container: #f1ecf2;
  --on-error-container: #93000a;
  --secondary-fixed-dim: #e2bdc8;
  --surface-container-highest: #e5e1e7;
  --on-secondary-fixed: #2b151d;
  --surface-dim: #ddd9de;
  --inverse-primary: #b5c8e2;
  --primary-container: #d1e4ff;
  --inverse-on-surface: #f4eff5;
  --tertiary-fixed-dim: #dac493;
  --on-secondary: #ffffff;
  --surface-container-high: #ebe7ec;
  --surface: #fdf8fd;
  --primary-fixed-dim: #b5c8e2;
  --error-container: #ffdad6;
  --secondary-container: #fcd6e2;
  --outline-variant: #c4c6cd;
  --surface-container-lowest: #ffffff;
`;

const darkVars = `
  --surface-bright: #39393c;
  --tertiary-fixed: #f7e0ad;
  --surface-tint: #b5c8e2;
  --on-primary-fixed: #081d30;
  --error: #ffb4ab;
  --on-tertiary: #3c2f0a;
  --on-primary: #1f3246;
  --on-secondary-container: #ffd9e4;
  --on-primary-fixed-variant: #36485e;
  --outline: #8e9097;
  --inverse-surface: #e5e1e7;
  --on-tertiary-fixed: #241a00;
  --on-surface: #e5e1e7;
  --on-error: #690005;
  --on-background: #e5e1e7;
  --tertiary-container: #54451f;
  --tertiary: #dac493;
  --on-surface-variant: #c4c6cd;
  --on-secondary-fixed-variant: #5a3f49;
  --secondary-fixed: #ffd9e4;
  --surface-variant: #44474c;
  --surface-container-low: #1c1b1f;
  --primary: #b5c8e2;
  --on-tertiary-fixed-variant: #54451f;
  --primary-fixed: #d1e4ff;
  --on-primary-container: #d1e4ff;
  --on-tertiary-container: #f7e0ad;
  --surface-container: #211f23;
  --on-error-container: #ffdad6;
  --secondary-fixed-dim: #e2bdc8;
  --surface-container-highest: #363438;
  --on-secondary-fixed: #2b151d;
  --surface-dim: #141316;
  --inverse-primary: #4e6076;
  --primary-container: #36485e;
  --inverse-on-surface: #313034;
  --tertiary-fixed-dim: #dac493;
  --on-secondary: #442933;
  --surface-container-high: #2b292d;
  --surface: #141316;
  --primary-fixed-dim: #b5c8e2;
  --error-container: #93000a;
  --secondary-container: #5a3f49;
  --outline-variant: #44474c;
  --surface-container-lowest: #0f0d11;
`;

// Insert root vars and dark vars just before @theme inline
css = css.replace('@theme inline {', ':root {' + lightVars + '}\n\n.dark {' + darkVars + '}\n\n@theme inline {');

// Replace static hex in @theme inline with var(--...)
css = css.replace(/--color-([a-z0-main-]+\-[a-z0-main-]+):\s*#[a-fA-F0-9]{6};/g, (match, p1) => {
  return '--color-' + p1 + ': var(--' + p1 + ');';
});

fs.writeFileSync('frontend/styles/globals.css', css);
