#!/usr/bin/env bash
# One-time root setup for the multi-user gateway. Run as:
#   sudo ./gateway/setup-root.sh
# Override the allowed networks if yours differ:
#   sudo LAN_SUBNET=10.49.56.0/23 VPN_SUBNET=172.18.0.0/16 ./gateway/setup-root.sh
#
# Does four things, each safe to re-run:
#   1. installs PostgreSQL (LiteLLM stores user keys and usage logs there)
#   2. creates the 'litellm' database and role, using the password that
#      './gateway.sh init' generated in .secrets/gateway.env
#   3. allows port 4000 (the gateway) from the LAN and the Cisco VPN pool
#      only — port 8000 (llama-server) is NOT opened; it should be bound
#      to 127.0.0.1 anyway
#   4. enables systemd "linger" for the repo owner, so their user services
#      (gateway + llama-server) start at boot without anyone logging in
set -euo pipefail

[[ "$(id -u)" -eq 0 ]] || { echo "Run with sudo: sudo $0" >&2; exit 1; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OWNER="$(stat -c %U "$REPO")"
SECRETS="$REPO/.secrets/gateway.env"
LAN_SUBNET="${LAN_SUBNET:-10.49.56.0/23}"
VPN_SUBNET="${VPN_SUBNET:-172.18.0.0/16}"

[[ -f "$SECRETS" ]] || { echo "$SECRETS not found — run './gateway.sh init' as $OWNER first." >&2; exit 1; }
PG_PASSWORD="$(grep '^PG_PASSWORD=' "$SECRETS" | cut -d= -f2-)"
[[ -n "$PG_PASSWORD" ]] || { echo "PG_PASSWORD missing from $SECRETS" >&2; exit 1; }

echo "== 1. PostgreSQL"
if ! command -v psql >/dev/null; then
  apt-get update -q
  apt-get install -y -q postgresql postgresql-client
fi
systemctl enable --now postgresql

echo "== 2. litellm database + role"
sudo -u postgres psql -v ON_ERROR_STOP=1 -q <<SQL
DO \$\$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'litellm') THEN
    CREATE ROLE litellm LOGIN PASSWORD '${PG_PASSWORD}';
  ELSE
    ALTER ROLE litellm WITH LOGIN PASSWORD '${PG_PASSWORD}';
  END IF;
END \$\$;
SQL
sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='litellm'" | grep -q 1 \
  || sudo -u postgres createdb -O litellm litellm
# Postgres listens on localhost only by default (Ubuntu) — left that way.

echo "== 3. firewall (ufw)"
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  ufw allow from "$LAN_SUBNET" to any port 4000 proto tcp comment 'LiteLLM gateway (LAN)'
  ufw allow from "$VPN_SUBNET" to any port 4000 proto tcp comment 'LiteLLM gateway (VPN)'
  if ufw status | grep -qE '^8000(/tcp)?\s+ALLOW'; then
    echo "NOTE: an existing ufw rule allows port 8000 (raw llama-server). Remove it with 'ufw status numbered' + 'ufw delete N'."
  fi
  ufw status verbose
else
  echo "ufw is not active — skipping. Port 4000 will be reachable from any network this box is on."
fi

echo "== 4. start at boot"
loginctl enable-linger "$OWNER"
echo "Linger enabled for $OWNER."

echo
echo "Done. Back as $OWNER: ./gateway.sh serve  (then ./gateway.sh add-user <name>)"
