// Standalone SVG export using the pinned Vega bundle, with no npm installation.
const fs = require('fs');
const path = require('path');
const [bundle, ...specPaths] = process.argv.slice(2);
const vega = require(path.resolve(bundle));
(async () => {
  for (const specPath of specPaths) {
    const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));
    const view = new vega.View(vega.parse(spec), {renderer: 'none'}).initialize();
    await view.runAsync();
    const svg = await view.toSVG();
    fs.writeFileSync(specPath.replace(/\.vega\.json$/, '.svg'), svg);
    view.finalize();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
