'use strict';
/**
 * 极简 WebSocket 服务端 (RFC6455), 仅支持文本消息 + ping/pong + close.
 * 无需任何第三方依赖, 适用于内网环境.
 */
const crypto = require('crypto');
const { EventEmitter } = require('events');

const GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
const OP = { CONT: 0, TEXT: 1, BIN: 2, CLOSE: 8, PING: 9, PONG: 10 };

function acceptKey(key) {
  return crypto.createHash('sha1').update(key + GUID).digest('base64');
}

class WSConn extends EventEmitter {
  constructor(socket) {
    super();
    this.socket = socket;
    this.buf = Buffer.alloc(0);
    this.frags = null;          // { op, chunks: [] }
    this.alive = true;          // 收到 pong 置 true
    this._emittedClose = false;
    socket.on('data', d => this._onData(d));
    const onEnd = () => this._destroy();
    socket.on('close', onEnd);
    socket.on('error', onEnd);
    socket.on('end', onEnd);
  }

  _onData(d) {
    this.buf = this.buf.length ? Buffer.concat([this.buf, d]) : d;
    for (;;) {
      const f = this._parse();
      if (!f) break;
      try { this._handle(f); } catch (e) { /* 单帧异常不致命 */ }
      if (this._emittedClose) return;
    }
  }

  _parse() {
    const b = this.buf;
    if (b.length < 2) return null;
    const fin = (b[0] & 0x80) !== 0;
    const op = b[0] & 0x0f;
    const masked = (b[1] & 0x80) !== 0;
    let len = b[1] & 0x7f;
    let off = 2;
    if (len === 126) {
      if (b.length < 4) return null;
      len = b.readUInt16BE(2); off = 4;
    } else if (len === 127) {
      if (b.length < 10) return null;
      const hi = b.readUInt32BE(2), lo = b.readUInt32BE(6);
      if (hi !== 0) return null;              // >4GB 不支持
      len = lo; off = 10;
    }
    let mask = null;
    if (masked) {
      if (b.length < off + 4) return null;
      mask = b.slice(off, off + 4); off += 4;
    }
    if (b.length < off + len) return null;
    let payload = b.slice(off, off + len);
    if (mask) {
      const out = Buffer.allocUnsafe(len);
      for (let i = 0; i < len; i++) out[i] = payload[i] ^ mask[i & 3];
      payload = out;
    }
    this.buf = b.slice(off + len);
    return { fin, op, payload };
  }

  _handle({ fin, op, payload }) {
    switch (op) {
      case OP.PING: this._send(OP.PONG, payload); return;
      case OP.PONG: this.alive = true; return;
      case OP.CLOSE: this.close(); return;
      case OP.TEXT: case OP.BIN: this.frags = { op, chunks: [payload] }; break;
      case OP.CONT: if (this.frags) this.frags.chunks.push(payload); else return; break;
      default: return;
    }
    if (fin && this.frags) {
      const data = Buffer.concat(this.frags.chunks);
      const isText = this.frags.op === OP.TEXT;
      this.frags = null;
      this.emit('message', isText ? data.toString('utf8') : data);
    }
  }

  _send(op, payload) {
    if (this._emittedClose) return;
    const len = payload.length;
    let header;
    if (len < 126) {
      header = Buffer.from([0x80 | op, len]);
    } else if (len < 65536) {
      header = Buffer.allocUnsafe(4);
      header[0] = 0x80 | op; header[1] = 126;
      header.writeUInt16BE(len, 2);
    } else {
      header = Buffer.allocUnsafe(10);
      header[0] = 0x80 | op; header[1] = 127;
      header.writeUInt32BE(0, 2); header.writeUInt32BE(len, 6);
    }
    try { this.socket.write(Buffer.concat([header, payload])); } catch (e) { this._destroy(); }
  }

  send(obj) {
    const str = typeof obj === 'string' ? obj : JSON.stringify(obj);
    this._send(OP.TEXT, Buffer.from(str, 'utf8'));
  }
  ping() { this._send(OP.PING, Buffer.alloc(0)); }
  close() {
    if (this._emittedClose) return;
    try { this._send(OP.CLOSE, Buffer.alloc(0)); } catch (e) {}
    try { this.socket.end(); } catch (e) {}
    this._destroy();
  }
  _destroy() {
    if (this._emittedClose) return;
    this._emittedClose = true;
    try { this.socket.destroy(); } catch (e) {}
    this.emit('close');
  }
}

/** 挂在 http.Server 的 upgrade 事件上 */
function attach(httpServer, onConn) {
  httpServer.on('upgrade', (req, socket) => {
    const key = req.headers['sec-websocket-key'];
    if ((req.headers.upgrade || '').toLowerCase() !== 'websocket' || !key) {
      socket.destroy();
      return;
    }
    socket.setNoDelay(true);
    socket.write(
      'HTTP/1.1 101 Switching Protocols\r\n' +
      'Upgrade: websocket\r\nConnection: Upgrade\r\n' +
      'Sec-WebSocket-Accept: ' + acceptKey(key) + '\r\n\r\n'
    );
    onConn(new WSConn(socket), req);
  });

  // 心跳: 每 25s ping 一次, 上次 ping 未回 pong 的连接直接断开
  const timer = setInterval(() => {
    for (const c of attach.conns) {
      if (!c.alive) { c._destroy(); continue; }
      c.alive = false;
      c.ping();
    }
  }, 25000);
  attach.conns = new Set();
  timer.unref();
  return attach.conns;
}

module.exports = { attach };
