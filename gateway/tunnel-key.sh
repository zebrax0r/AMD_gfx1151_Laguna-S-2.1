#!/usr/bin/env bash
# Manage who can open an SSH tunnel to the gateway (see setup-tunnel-root.sh).
#   sudo ./gateway/tunnel-key.sh add <name> '<ssh public key>'
#   sudo ./gateway/tunnel-key.sh remove <name>
#   sudo ./gateway/tunnel-key.sh list
# <name> should match their LiteLLM key name (./gateway.sh add-user <name>)
# so one name identifies a person in both places. Revoking someone fully =
# `tunnel-key.sh remove <name>` + `./gateway.sh revoke-user <name>`.
set -euo pipefail

[[ "$(id -u)" -eq 0 ]] || { echo "Run with sudo: sudo $0 $*" >&2; exit 1; }
KEYS=/etc/ssh/llm-tunnel/authorized_keys
[[ -f "$KEYS" ]] || { echo "$KEYS missing — run sudo ./gateway/setup-tunnel-root.sh first." >&2; exit 1; }
# Per-key options repeat the sshd Match-block limits, so a key is still
# confined even if the drop-in were ever removed.
OPTS='restrict,port-forwarding,permitopen="127.0.0.1:4000",command="/usr/sbin/nologin"'

case "${1:-}" in
  add)
    name="${2:-}"; key="${3:-}"
    [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Usage: $0 add <name> '<ssh public key>' (name: letters, digits, . _ -)" >&2; exit 1; }
    tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
    printf '%s\n' "$key" > "$tmp"
    ssh-keygen -l -f "$tmp" >/dev/null 2>&1 || { echo "That doesn't look like a valid SSH public key (expected e.g. 'ssh-ed25519 AAAA... comment')." >&2; exit 1; }
    grep -q " llm-user=${name}\$" "$KEYS" && { echo "'$name' already has a key — remove it first." >&2; exit 1; }
    # Keep only "type base64"; replace their comment with our own tag.
    read -r ktype kdata _ <<<"$key"
    printf '%s %s %s llm-user=%s\n' "$OPTS" "$ktype" "$kdata" "$name" >> "$KEYS"
    echo "Added tunnel key for '$name' ($(ssh-keygen -l -f "$tmp" | awk '{print $2}'))."
    ;;
  remove)
    name="${2:-}"; [[ -n "$name" ]] || { echo "Usage: $0 remove <name>" >&2; exit 1; }
    grep -q " llm-user=${name}\$" "$KEYS" || { echo "No tunnel key for '$name'." >&2; exit 1; }
    sed -i "/ llm-user=${name}\$/d" "$KEYS"
    echo "Removed tunnel key for '$name'. Existing open tunnels stay up until they disconnect;"
    echo "to cut them now: sudo pkill -u llm-tunnel sshd   (drops ALL tunnel users' sessions)"
    ;;
  list)
    grep -oE 'llm-user=[A-Za-z0-9._-]+' "$KEYS" | cut -d= -f2 || echo "(no tunnel users)"
    ;;
  *)
    sed -n '2,8p' "$0"; exit 1 ;;
esac
