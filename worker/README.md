# Star → 自动构建 触发器

你在 GitHub 仓库页点 ⭐（star/unstar 都行，只有 `action:created` 即"点成星"才算），Cloudflare Worker 收到 webhook 后调 `repository_dispatch`，触发 `release.yml` 在 macOS runner 上打包 `斗地主.app` 并发新 Release。**白名单限制：只有仓库所有者（默认）或 `ALLOWED` 变量里列出的 GitHub 登录名的 star 才触发，别人的 star 一律忽略。**

## 部署（一次性，约 5 分钟）

### 1. 部署 Worker
1. 注册 [Cloudflare](https://dash.cloudflare.com/) → 左侧 **Workers & Pages** → **Create** → **Create Worker** → Deploy
2. 点 **Edit code**，把 `worker/star-hook.js` 全部内容粘进去 → **Deploy**
3. 记下 Worker URL：`https://<name>.<subdomain>.workers.dev`

### 2. 配两个 secret
Worker 页 → **Settings** → **Variables and Secrets** → **Add**：

| 变量 | 值 |
|---|---|
| `GH_TOKEN` | GitHub Personal Access Token (classic)，勾选 `repo` + `workflow` 两个 scope |
| `WEBHOOK_SECRET` | 随便一串长随机字符串（下面 GitHub webhook 也要填同一个） |
| `GH_REPO` | `你的用户名/doudizhu`（可选，默认取 payload 里的仓库名） |
| `ALLOWED` | 逗号分隔的 GitHub 登录名白名单（可选，默认只放行仓库所有者） |

### 3. GitHub 仓库配 webhook
仓库 → **Settings** → **Webhooks** → **Add webhook**：

- **Payload URL**：`https://<name>.<subdomain>.workers.dev`
- **Content type**：`application/json`
- **Secret**：上面那个 `WEBHOOK_SECRET`
- **Which events**：选 **Let me select individual events** → 只勾 **Watching**（别选 All）

保存。GitHub 会发个 ping 请求，Worker 返回 `ok`（对非 watch 事件忽略，ping 不算 star）。

## 验证
1. 用小号或朋友的号给仓库点 ⭐
2. Worker 日志（Workers → 你的 worker → Logs）应看到 `dispatched for <who>`
3. 仓库 Actions 页出现新的 `Release` run，构建完 Releases 里多一个 `v*` 版本

## 安全/注意
- **HMAC-SHA256 签名校验**：Worker 会校验 GitHub webhook 签名，伪造请求会被拒（401）
- **只对 `action: created` 触发**：unstar（取消 star）也发 webhook，忽略
- **60s 冷却**：同一用户反复 star/unstar 只触发一次
- **Worker 免费额度**：每天 10 万次请求，这个项目绰绰有余
- 如果 star 太多触发频繁构建，可以在 workflow 里加 `concurrency` 节流

## 不想用 Cloudflare？
同一逻辑可搬到 Vercel Serverless / Deno Deploy / 自建小服务，只要把 `fetch` handler 改成对应平台的入口即可。
