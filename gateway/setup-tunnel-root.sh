#!/usr/bin/env bash
# One-time root setup for SSH-tunnel access to the gateway. Run as:
#   sudo ./gateway/setup-tunnel-root.sh
#
# Why: on this network the Cisco VPN only lets port 22 through to this box
# (tested 2026-10-06: 3000/4000/5000/7860/8001/8080/8443/8888/9000/9090/11434
# all dropped upstream, before reaching the box). Users reach the gateway
# by forwarding a local port over SSH to 127.0.0.1:4000 instead.
#
# Creates one shared account, llm-tunnel, that can do exactly one thing:
# forward to 127.0.0.1:4000. No password, no shell, no TTY, no other
# forwarding destinations, no agent/X11 forwarding. Each user is one line in
# /etc/ssh/llm-tunnel/authorized_keys (see gateway/tunnel-key.sh), so access
# is revoked by removing their key — and their LiteLLM key still controls
# and meters what they can do once connected. Safe to re-run.
set -euo pipefail

[[ "$(id -u)" -eq 0 ]] || { echo "Run with sudo: sudo $0" >&2; exit 1; }
TUNNEL_USER=llm-tunnel
KEYS_DIR=/etc/ssh/llm-tunnel
DROPIN=/etc/ssh/sshd_config.d/50-llm-tunnel.conf

echo "== account"
if ! getent passwd "$TUNNEL_USER" >/dev/null; then
  useradd --system --create-home --home-dir /var/lib/llm-tunnel \
    --shell /usr/sbin/nologin --comment "SSH tunnel to LiteLLM gateway only" "$TUNNEL_USER"
fi
passwd -l "$TUNNEL_USER" >/dev/null   # no password login, ever

echo "== key file"
install -d -o root -g root -m 755 "$KEYS_DIR"
[[ -f "$KEYS_DIR/authorized_keys" ]] || install -o root -g root -m 644 /dev/null "$KEYS_DIR/authorized_keys"

echo "== sshd restrictions"
cat > "$DROPIN" <<EOF
# Managed by gateway/setup-tunnel-root.sh — tunnel-only account for the
# LiteLLM gateway. Keys live in ${KEYS_DIR}/authorized_keys (root-owned, so
# the account itself can't add keys).
Match User ${TUNNEL_USER}
    AuthorizedKeysFile ${KEYS_DIR}/authorized_keys
    PasswordAuthentication no
    KbdInteractiveAuthentication no
    AllowTcpForwarding local
    PermitOpen 127.0.0.1:4000
    PermitListen none
    AllowAgentForwarding no
    AllowStreamLocalForwarding no
    X11Forwarding no
    PermitTTY no
    PermitTunnel no
    ForceCommand /usr/sbin/nologin
EOF
chmod 644 "$DROPIN"

sshd -t || { echo "sshd config test FAILED — removing $DROPIN, sshd untouched." >&2; rm -f "$DROPIN"; exit 1; }
systemctl reload ssh
echo
echo "Done. Add users with: sudo ./gateway/tunnel-key.sh add <name> '<their ssh public key>'"
