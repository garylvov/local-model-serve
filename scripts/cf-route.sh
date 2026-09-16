#!/usr/bin/env bash
# Add/update ONE ingress rule on the existing remotely-managed Cloudflare Tunnel:
#   <hostname> -> <service>   (default llm.garylvov.com -> http://<gateway-host>:4000)
# preserving every other rule verbatim, plus the proxied DNS CNAME if missing.
#
# DEFAULT IS DRY-RUN: only GETs are made; prints the before/after ingress diff and
# the DNS action. `--apply` performs the PUT (tunnel configuration) and POST (DNS).
#
# Usage: scripts/cf-route.sh [--hostname llm.garylvov.com] [--service http://gpu2260:4000]
#                            [--tunnel-id ID] [--account-id ID] [--zone garylvov.com] [--apply]
# Env:   CF_API_TOKEN_FILE (default ~/.config/slurm-dash/cloudflare-api-token)
#
# The token is only ever fed to curl through a stdin config (-K -), never argv or output.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOSTNAME_FQDN="llm.garylvov.com"
ZONE_NAME="garylvov.com"
SERVICE=""
TUNNEL_ID="${CF_TUNNEL_ID:-}"
ACCOUNT_ID="${CF_ACCOUNT_ID:-}"
ANCHOR_HOST="${CF_ANCHOR_HOST:-ccv.garylvov.com}"   # used to find the right tunnel
APPLY=0
TOKEN_FILE="${CF_API_TOKEN_FILE:-$HOME/.config/slurm-dash/cloudflare-api-token}"
API="https://api.cloudflare.com/client/v4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hostname) HOSTNAME_FQDN="$2"; shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --tunnel-id) TUNNEL_ID="$2"; shift 2 ;;
    --account-id) ACCOUNT_ID="$2"; shift 2 ;;
    --zone) ZONE_NAME="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$SERVICE" ]]; then
  gw="$(ls "$ROOT"/registry/gateway-*.json 2>/dev/null | head -1 || true)"
  if [[ -n "$gw" ]]; then SERVICE="$(jq -r .url "$gw")"; else SERVICE="http://$(hostname -s):4000"; fi
fi

[[ -r "$TOKEN_FILE" ]] || { echo "token file not readable: $TOKEN_FILE" >&2; exit 1; }
mode="$(stat -c %a "$TOKEN_FILE")"
[[ "$mode" == 600 || "$mode" == 400 ]] || { echo "$TOKEN_FILE must be mode 0600" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq required" >&2; exit 1; }

cf() { # cf METHOD PATH [JSON-BODY]
  local method="$1" path="$2" body="${3:-}"
  if [[ "$method" != GET && "$APPLY" != 1 ]]; then
    echo "internal error: refusing $method in dry-run" >&2; exit 3
  fi
  local args=(-sS --fail-with-body -X "$method" "$API$path" -H "Content-Type: application/json")
  [[ -n "$body" ]] && args+=(--data "$body")
  { printf 'header = "Authorization: Bearer %s"\n' "$(tr -d '\r\n' < "$TOKEN_FILE")"; } | curl -K - "${args[@]}"
}

ok() { jq -e '.success == true' >/dev/null <<<"$1" || { echo "API error: $(jq -c '.errors' <<<"$1")" >&2; exit 1; }; }

# ---- discover account, tunnel, zone (GET only)
zr="$(cf GET "/zones?name=$ZONE_NAME")"; ok "$zr"
ZONE_ID="$(jq -r '.result[0].id // empty' <<<"$zr")"
[[ -n "$ZONE_ID" ]] || { echo "zone $ZONE_NAME not visible to token" >&2; exit 1; }
if [[ -z "$ACCOUNT_ID" ]]; then
  # Zone-scoped tokens often cannot list accounts; the zone carries its account id.
  ACCOUNT_ID="$(jq -r '.result[0].account.id // empty' <<<"$zr")"
  [[ -n "$ACCOUNT_ID" ]] || { echo "cannot determine account id; pass --account-id" >&2; exit 1; }
fi

if [[ -z "$TUNNEL_ID" ]]; then
  r="$(cf GET "/accounts/$ACCOUNT_ID/cfd_tunnel?is_deleted=false&per_page=100")"; ok "$r"
  for id in $(jq -r '.result[] | select(.config_src == "cloudflare") | .id' <<<"$r"); do
    c="$(cf GET "/accounts/$ACCOUNT_ID/cfd_tunnel/$id/configurations")"
    if jq -e --arg h "$ANCHOR_HOST" '.result.config.ingress // [] | any(.hostname == $h)' >/dev/null <<<"$c"; then
      TUNNEL_ID="$id"; break
    fi
  done
  [[ -n "$TUNNEL_ID" ]] || { echo "no remotely-managed tunnel routes $ANCHOR_HOST; pass --tunnel-id" >&2; exit 1; }
fi

tun="$(cf GET "/accounts/$ACCOUNT_ID/cfd_tunnel/$TUNNEL_ID")"; ok "$tun"
echo "tunnel: $(jq -r '.result | "\(.name) id=\(.id) status=\(.status) connections=\(.connections | length)"' <<<"$tun")"

cur="$(cf GET "/accounts/$ACCOUNT_ID/cfd_tunnel/$TUNNEL_ID/configurations")"; ok "$cur"
version="$(jq -r '.result.version' <<<"$cur")"
before="$(jq '.result.config' <<<"$cur")"

# Update in place if the hostname exists (keeping its other keys, e.g. originRequest);
# otherwise insert just before the first hostname-less catch-all rule.
after="$(jq --arg h "$HOSTNAME_FQDN" --arg s "$SERVICE" '
  .ingress as $ing
  | if ($ing | any(.hostname == $h)) then
      .ingress = [ $ing[] | if .hostname == $h then .service = $s else . end ]
    else
      ($ing | map(has("hostname") | not) | index(true)) as $ci
      | if $ci == null then error("ingress has no catch-all rule; refusing")
        else .ingress = ($ing[0:$ci] + [{"hostname": $h, "service": $s, "originRequest": {}}] + $ing[$ci:])
        end
    end' <<<"$before")"

# Safety: every rule other than ours must be byte-identical and in the same order.
jq -e --arg h "$HOSTNAME_FQDN" --argjson b "$before" '
  ([.ingress[] | select(.hostname != $h)] == [$b.ingress[] | select(.hostname != $h)])
  and ((del(.ingress)) == ($b | del(.ingress)))' >/dev/null <<<"$after" \
  || { echo "internal error: other rules would change; refusing" >&2; exit 3; }

echo "config version: $version"
echo "---- ingress diff (before -> after)"
diff -u --label before --label after <(jq . <<<"$before") <(jq . <<<"$after") || true

# ---- DNS
dr="$(cf GET "/zones/$ZONE_ID/dns_records?name=$HOSTNAME_FQDN")"; ok "$dr"
target="$TUNNEL_ID.cfargotunnel.com"
existing="$(jq -c '.result[] | {type, name, content, proxied}' <<<"$dr")"
dns_action="none"
if [[ -z "$existing" ]]; then
  dns_action="create"
  echo "---- DNS: would CREATE CNAME $HOSTNAME_FQDN -> <tunnel-id>.cfargotunnel.com (proxied)"
elif jq -e --arg t "$target" 'select(.type == "CNAME" and .content == $t)' >/dev/null <<<"$existing"; then
  echo "---- DNS: CNAME already points at this tunnel; no change"
else
  dns_action="conflict"
  echo "---- DNS: $HOSTNAME_FQDN already has a different record: $(jq -c '{type, proxied}' <<<"$existing"); will NOT touch it" >&2
fi

if [[ "$APPLY" != 1 ]]; then
  echo "---- dry-run: no changes made (rerun with --apply after operator approval)"
  exit 0
fi

# ---- apply (operator-approved only)
body="$(jq -c '{config: .}' <<<"$after")"
r="$(cf PUT "/accounts/$ACCOUNT_ID/cfd_tunnel/$TUNNEL_ID/configurations" "$body")"; ok "$r"
echo "tunnel configuration updated to version $(jq -r '.result.version' <<<"$r")"
if [[ "$dns_action" == create ]]; then
  r="$(cf POST "/zones/$ZONE_ID/dns_records" "$(jq -nc --arg n "$HOSTNAME_FQDN" --arg c "$target" \
    '{type: "CNAME", name: $n, content: $c, proxied: true, comment: "local-model-serve gateway"}')")"; ok "$r"
  echo "DNS CNAME created"
fi
