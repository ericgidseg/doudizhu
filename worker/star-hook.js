/**
 * Cloudflare Worker: GitHub star webhook -> repository_dispatch 触发构建
 *
 * 部署:
 *   1. 注册 Cloudflare 账号 → Workers → Create Worker → 把本文件全部粘进编辑器 → Deploy
 *   2. Worker 设置 → Variables → 加两个 secret:
 *        GH_TOKEN   : GitHub Personal Access Token (classic, scope: repo + workflow)
 *        WEBHOOK_SECRET : 随便一串长随机字符串, 下面配 GitHub webhook 时要用同一个
 *        GH_REPO    : "owner/repo" (例 "ericgidseg/doudizhu"), 也可不加用默认值
 *   3. GitHub 仓库 → Settings → Webhooks → Add webhook:
 *        Payload URL : https://<你的worker>.workers.dev
 *        Content type: application/json
 *        Secret      : 上面的 WEBHOOK_SECRET
 *        Events      : 只选 "Watch" (即 star; 别选 star 仓库的其它事件)
 *   4. 只有仓库所有者(或 ALLOWED 白名单)点 star 才触发构建; 别人的 star 直接忽略.
 *
 * 防刷: 同一 starrer 60s 内多次 star/unstar 只算一次 (GitHub 也会重发, 这里再挡一层).
 */

const COOLDOWN_MS = 60_000;
const recent = new Map();   // starrer login -> last trigger ts (Worker 实例内, 重启清)

export default {
  async fetch(req, env) {
    if (req.method !== 'POST') return new Response('ok', { status: 200 });

    const sig = req.headers.get('X-Hub-Signature-256');
    const event = req.headers.get('X-GitHub-Event');
    if (!sig || event !== 'watch' && event !== 'star') {
      return new Response('bad request', { status: 400 });
    }

    const body = await req.text();
    if (!(await verify(sig, body, env.WEBHOOK_SECRET))) {
      return new Response('bad signature', { status: 401 });
    }

    let payload;
    try { payload = JSON.parse(body); } catch { return new Response('bad json', { status: 400 }); }
    const action = payload.action;                    // 'created' = 点了 star; 'deleted' = 取消
    const who = payload.sender && payload.sender.login;

    /* 只对 "新增 star" 触发; unstar 也发 webhook, 忽略 */
    if (action !== 'created') return new Response('ignored (not a star)', { status: 200 });

    /* 白名单: 只有仓库所有者(或 ALLOWED 列表里的人)的 star 才触发构建 */
    const allowed = (env.ALLOWED || (payload.repository && payload.repository.owner && payload.repository.owner.login) || '')
      .split(',').map(s => s.trim()).filter(Boolean);
    if (allowed.length && !allowed.includes(who)) {
      return new Response(`ignored (${who} not in allowlist)`, { status: 200 });
    }

    /* 冷却 */
    const now = Date.now();
    const last = recent.get(who) || 0;
    if (now - last < COOLDOWN_MS) return new Response(`cooldown for ${who}`, { status: 200 });
    recent.set(who, now);

    const repo = env.GH_REPO || (payload.repository && payload.repository.full_name);
    if (!repo) return new Response('no repo', { status: 400 });

    const gh = await fetch(`https://api.github.com/repos/${repo}/dispatches`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${env.GH_TOKEN}`,
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'doudizhu-star-hook',
        'X-GitHub-Api-Version': '2022-11-28',
      },
      body: JSON.stringify({ event_type: 'star' }),
    });
    if (!gh.ok && gh.status !== 204) {
      const t = await gh.text();
      return new Response(`github ${gh.status}: ${t.slice(0, 200)}`, { status: 502 });
    }
    return new Response(`dispatched for ${who}`, { status: 200 });
  },
};

/* HMAC-SHA256 验证 GitHub webhook 签名 */
async function verify(sig, body, secret) {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    'raw', enc.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const mac = await crypto.subtle.sign('HMAC', key, enc.encode(body));
  const hex = [...new Uint8Array(mac)].map(b => b.toString(16).padStart(2, '0')).join('');
  const expect = `sha256=${hex}`;
  return expect.length === sig.length && timingSafeEqual(expect, sig);
}
function timingSafeEqual(a, b) {
  if (a.length !== b.length) return false;
  let r = 0;
  for (let i = 0; i < a.length; i++) r |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return r === 0;
}
