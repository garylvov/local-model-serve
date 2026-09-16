#!/usr/bin/env bash
# Plan the Cloudflare setup that lets ONE off-cluster machine be a gateway peer:
#   <machine>.llm-peers.<CF_ZONE>  ->  that machine's own cloudflared tunnel -> http://127.0.0.1:8080
# protected by a Cloudflare Access application that accepts one Access service token
# (the gateway sends CF-Access-Client-Id/Secret from ~/.config/local-model-serve/cf-access.env).
#
# DEFAULT IS DRY-RUN: it makes read-only GETs to discover the zone/account and to check what
# already exists, then PRINTS every write it would make. `--apply` exists but is intentionally
# refused unless LMS_CF_APPLY_OK=i-have-operator-approval is also set.
#
# Usage: scripts/cf-machine.sh <machine> [--port 8080] [--zone example.com] [--apply]
# Config: CF_ZONE, CF_API_TOKEN_FILE, LLM_PUBLIC_URL (config.env or environment)
#
# Each machine gets its OWN tunnel and token: a second connector on an existing token would
# receive a share of that tunnel's traffic and break whatever else it serves.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# site settings (CF_ZONE, LLM_PUBLIC_URL, CF_API_TOKEN_FILE ...) from config.env; env vars win
for cfg in "${LLM_SITE_CONFIG:-}" "$ROOT/config.env" "$HOME/.config/local-model-serve/config.env"; do
  [[ -n "$cfg" && -r "$cfg" ]] || continue
  while IFS='=' read -r k v; do [[ -n "${!k:-}" ]] || export "$k=${v/#\~/$HOME}"; done \
    < <(grep -E '^[A-Z_][A-Z0-9_]*=' "$cfg" | sed -E 's/[[:space:]]+#.*$//')
  break
done

MACHINE="${1:-}"; shift || true
[[ -n "$MACHINE" && "$MACHINE" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "usage: cf-machine.sh <machine> [--apply]" >&2; exit 2; }
ZONE_NAME="${CF_ZONE:-}"; PORT=8080; APPLY=0
SUBDOMAIN_SUFFIX="llm-peers"
TOKEN_FILE="${CF_API_TOKEN_FILE:-$HOME/.config/local-model-serve/cloudflare-api-token}"
API=https://api.cloudflare.com/client/v4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --zone) ZONE_NAME="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
HOSTNAME_FQDN="$MACHINE.$SUBDOMAIN_SUFFIX.$ZONE_NAME"
TUNNEL_NAME="llm-peer-$MACHINE"
ACCESS_APP_NAME="llm peer $MACHINE"
SERVICE_TOKEN_NAME="llm-gateway"

[[ -r "$TOKEN_FILE" ]] || { echo "token file not readable: $TOKEN_FILE" >&2; exit 1; }
[[ "$(stat -c %a "$TOKEN_FILE")" =~ ^(600|400)$ ]] || { echo "$TOKEN_FILE must be mode 0600" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq required" >&2; exit 1; }

cf() { # cf GET <path>   (only GETs are ever executed by this script)
  [[ "$1" == GET ]] || { echo "internal: cf() only performs GETs" >&2; exit 3; }
  { printf 'header = "Authorization: Bearer %s"\n' "$(tr -d '\r\n' < "$TOKEN_FILE")"; } \
    | curl -K - -sS --fail-with-body "$API$2"
}
ok() { jq -e '.success == true' >/dev/null <<<"$1" || { echo "API error: $(jq -c .errors <<<"$1")" >&2; exit 1; }; }

zr="$(cf GET "/zones?name=$ZONE_NAME")"; ok "$zr"
ZONE_ID="$(jq -r '.result[0].id' <<<"$zr")"
ACCOUNT_ID="$(jq -r '.result[0].account.id' <<<"$zr")"
echo "zone $ZONE_NAME found; account and zone ids resolved (not printed)"

existing_tunnel="$(cf GET "/accounts/$ACCOUNT_ID/cfd_tunnel?name=$TUNNEL_NAME&is_deleted=false" | jq -r '.result[0].id // empty')"
existing_dns="$(cf GET "/zones/$ZONE_ID/dns_records?name=$HOSTNAME_FQDN" | jq -r '.result[0].id // empty')"
existing_app="$(cf GET "/accounts/$ACCOUNT_ID/access/apps" | jq -r --arg d "$HOSTNAME_FQDN" '.result[]? | select(.domain == $d) | .id' | head -1)"
existing_st="$(cf GET "/accounts/$ACCOUNT_ID/access/service_tokens" | jq -r --arg n "$SERVICE_TOKEN_NAME" '.result[]? | select(.name == $n) | .id' | head -1)"

echo "current state:"
echo "  tunnel $TUNNEL_NAME        : ${existing_tunnel:+exists}${existing_tunnel:-missing}"
echo "  DNS $HOSTNAME_FQDN         : ${existing_dns:+exists}${existing_dns:-missing}"
echo "  Access app for that host   : ${existing_app:+exists}${existing_app:-missing}"
echo "  Access service token '$SERVICE_TOKEN_NAME': ${existing_st:+exists}${existing_st:-missing}"

cat <<PLAN

planned writes (NOT executed in dry-run):
1. POST $API/accounts/<account>/cfd_tunnel
   {"name": "$TUNNEL_NAME", "config_src": "cloudflare"}
   -> a NEW tunnel, separate from any existing one. Never copy an existing token.
2. PUT  $API/accounts/<account>/cfd_tunnel/<new-tunnel>/configurations
   {"config": {"ingress": [{"hostname": "$HOSTNAME_FQDN", "service": "http://127.0.0.1:$PORT"},
                            {"service": "http_status:404"}]}}
3. POST $API/zones/<zone>/dns_records
   {"type": "CNAME", "name": "$HOSTNAME_FQDN", "content": "<new-tunnel>.cfargotunnel.com", "proxied": true}
4. POST $API/accounts/<account>/access/service_tokens        (only if missing)
   {"name": "$SERVICE_TOKEN_NAME"}   -> client_secret is returned ONCE; store it as
   ~/.config/local-model-serve/cf-access.env (0600) on the GATEWAY host only:
   CF_ACCESS_CLIENT_ID=... / CF_ACCESS_CLIENT_SECRET=...
5. POST $API/accounts/<account>/access/apps
   {"name": "$ACCESS_APP_NAME", "domain": "$HOSTNAME_FQDN", "type": "self_hosted",
    "session_duration": "24h", "policies": [{"name": "gateway service token", "decision": "non_identity",
    "include": [{"service_token": {"token_id": "<service-token-id>"}}]}]}
   -> deny-by-default for everyone else; no email/OTP policy, no Bypass, no Everyone.
6. GET  $API/accounts/<account>/cfd_tunnel/<new-tunnel>/token
   -> install on THAT machine as ~/.config/local-model-serve/tunnel-token (0600), then:
      llm tunnel up        # cloudflared --protocol quic, falls back to http2
      # in ~/.config/local-model-serve/auth.env on that machine:
      LLM_GATEWAY_URL=$LLM_PUBLIC_URL
      LLM_PEER_URL=https://$HOSTNAME_FQDN
      LLM_PEER_CF_ACCESS=true
      llm serve            # starts the router and heartbeats the peer URL to the gateway

PLAN

if (( APPLY )); then
  [[ "${LMS_CF_APPLY_OK:-}" == i-have-operator-approval ]] || {
    echo "--apply refused: set LMS_CF_APPLY_OK=i-have-operator-approval (operator action only)" >&2; exit 4; }
  echo "apply path intentionally not implemented in this revision: run the calls above by hand" >&2
  exit 5
fi
echo "dry-run only: no Cloudflare writes were made."
