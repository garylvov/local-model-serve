#!/usr/bin/env node
// Minimal dependency-free TCP relay: LISTEN_HOST:LISTEN_PORT -> TARGET_HOST:TARGET_PORT.
//
// Why a relay at all: the Agor native daemon hard-codes DAEMON_HOST=127.0.0.1
// (imprint.integrations.agor.native_daemon, the `env` dict it builds for the node
// child), so it is reachable only from inside its own compute node. Nothing about
// the daemon's sealed config is changed here; the relay sits beside it.
//
// Raw byte forwarding, so HTTP, WebSocket upgrades and socket.io all pass through.
// No TLS and no auth: the transport is private (cluster network / loopback) and the
// authorization boundary stays where it already is -- Agor's own login, plus
// Cloudflare Access in front of the public hostname.
//
// usage: node tcp-relay.mjs <listen-host> <listen-port> <target-host> <target-port>
import net from 'node:net';

const [listenHost, listenPortRaw, targetHost, targetPortRaw] = process.argv.slice(2);
if (!listenHost || !listenPortRaw || !targetHost || !targetPortRaw) {
  console.error('usage: tcp-relay.mjs <listen-host> <listen-port> <target-host> <target-port>');
  process.exit(2);
}
const listenPort = Number(listenPortRaw);
const targetPort = Number(targetPortRaw);
for (const [name, value] of [['listen', listenPort], ['target', targetPort]]) {
  if (!Number.isInteger(value) || value < 1024 || value > 65535) {
    console.error(`${name} port out of range: ${value}`);
    process.exit(2);
  }
}

let live = 0;
const server = net.createServer((client) => {
  live += 1;
  const upstream = net.connect(targetPort, targetHost);
  // A refusal upstream is visible as a closed client socket; never coerced to a
  // benign empty response.
  const shut = (why) => {
    if (!client.destroyed) client.destroy();
    if (!upstream.destroyed) upstream.destroy();
    if (why) console.error(`[relay] ${why}`);
  };
  client.on('error', (e) => shut(`client: ${e.message}`));
  upstream.on('error', (e) => shut(`upstream ${targetHost}:${targetPort}: ${e.message}`));
  client.on('close', () => { live -= 1; shut(); });
  upstream.on('close', () => shut());
  upstream.on('connect', () => {
    client.pipe(upstream);
    upstream.pipe(client);
  });
});
server.on('error', (e) => {
  console.error(`[relay] listen ${listenHost}:${listenPort}: ${e.message}`);
  process.exit(1);
});
server.listen(listenPort, listenHost, () => {
  console.log(`[relay] ${listenHost}:${listenPort} -> ${targetHost}:${targetPort} (pid ${process.pid})`);
});
setInterval(() => {
  console.log(`[relay] ${new Date().toISOString()} live=${live}`);
}, 300000).unref?.();
for (const sig of ['SIGTERM', 'SIGINT']) {
  process.on(sig, () => { server.close(() => process.exit(0)); });
}
