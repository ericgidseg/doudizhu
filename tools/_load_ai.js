'use strict';
/* 统一加载 AI 模块: 支持 new|baseline 别名, 也支持任意路径的 AI 快照
   (归档文件里的 require('./cards.js') 一律指向项目根). */
const path = require('path');
const fs = require('fs');
const vm = require('vm');
const Module = require('module');
const C = require(path.join(__dirname, '..', 'cards.js'));
const AI = {
  new: require(path.join(__dirname, '..', 'ai.js')),
  baseline: require(path.join(__dirname, '..', 'ai_baseline.js')),
};
function loadAI(spec) {
  if (AI[spec]) return AI[spec];
  if (!spec) return null;
  if (/[\\/]/.test(spec) || spec.endsWith('.js')) {
    const p = path.isAbsolute(spec) ? spec : path.resolve(process.cwd(), spec);
    const src = fs.readFileSync(p, 'utf8');
    const mod = new Module(p, null);
    mod.filename = p;
    mod.paths = Module._nodeModulePaths(path.dirname(p));
    const customRequire = (id) => {
      if (id === './cards.js' || id === '../cards.js') return C;
      return Module.createRequire(p)(id);
    };
    const fn = vm.runInThisContext(Module.wrap(src), { filename: p });
    fn.call(mod.exports, mod.exports, customRequire, mod, p, path.dirname(p));
    return mod.exports;
  }
  return null;
}
module.exports = { loadAI, C };
