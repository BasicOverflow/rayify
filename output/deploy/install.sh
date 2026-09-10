#!/usr/bin/env bash
# Run on a Proxmox node: create LXC, install Lexis Markets supervisor (Python), start systemd.
#
# This is NOT the DeepSeek Harness path — no npm, no Node, no NPM proxy registration.
# Runtime: uv-managed CPython + systemd unit talking to Ray / MinIO / Postgres over LAN.
#
# One-liner (public app repo):
#   bash -c "$(curl -fsSL http://10.0.0.52:3000/admin/lexis-markets/raw/branch/main/deploy/install.sh)"
#
# Cluster .env still lives in private admin/rayify_real_infra_creds — put a host-only
# token file at /root/lexis-markets.secrets.env (GITEA_TOKEN=...) so install can fetch it.
set -euo pipefail

REPO_RAW_BASE="${REPO_RAW_BASE:-http://10.0.0.52:3000/admin/lexis-markets/raw/branch/main}"
REPO_CLONE_URL="${REPO_CLONE_URL:-http://10.0.0.52:3000/admin/lexis-markets.git}"
WORK_DIR="${WORK_DIR:-/tmp/lexis-markets-deploy}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing command: $1" >&2; exit 1; }
}

need_cmd pct
need_cmd curl

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run as root on a Proxmox node" >&2
  exit 1
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

# Host-only secrets (never committed). Used only to pull private rayify_real_infra_creds.
if [[ -n "${SECRETS_FILE:-}" && -f "$SECRETS_FILE" ]]; then
  set -a; # shellcheck disable=SC1090
  source "$SECRETS_FILE"; set +a
elif [[ -f /root/lexis-markets.secrets.env ]]; then
  set -a; # shellcheck disable=SC1091
  source /root/lexis-markets.secrets.env; set +a
fi

if ! command -v git >/dev/null 2>&1; then
  echo "install: installing git on Proxmox host"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq git
fi

echo "install: cloning public app repo"
if ! git clone --depth 1 "$REPO_CLONE_URL" "$WORK_DIR"; then
  echo "install: git clone failed for ${REPO_CLONE_URL}" >&2
  exit 1
fi
find "$WORK_DIR" -type f -exec sed -i 's/\r$//' {} +

__OVERRIDES=$(mktemp)
for v in CTID CT_HOSTNAME CT_IP CT_GW CT_CIDR BRIDGE STORAGE TEMPLATE_STORAGE OSTEMPLATE MEMORY CORES DISK_GB UNPRIVILEGED FEATURES START_ON_BOOT HA_GROUP APP_ROOT APP_USER STATE_DIR ENV_FILE CREDS_REPO_RAW GITEA_TOKEN SECRETS_FILE; do
  if [[ -n "${!v+x}" ]]; then
    printf '%s=%q\n' "$v" "${!v}" >>"$__OVERRIDES"
  fi
done
# shellcheck disable=SC1091
source "${WORK_DIR}/deploy/defaults.env"
# shellcheck disable=SC1090
source "$__OVERRIDES"
rm -f "$__OVERRIDES"

echo "install: CTID=${CTID} hostname=${CT_HOSTNAME} ip=${CT_IP}/${CT_CIDR} storage=${STORAGE}"

if pct status "$CTID" >/dev/null 2>&1; then
  echo "install: CT ${CTID} already exists — skipping pct create (will re-bootstrap)"
  pct set "$CTID" --hostname "$CT_HOSTNAME" || true
else
  TEMPLATE_PATH="${TEMPLATE_STORAGE}:vztmpl/${OSTEMPLATE}"
  if ! pveam list "$TEMPLATE_STORAGE" 2>/dev/null | grep -q "$OSTEMPLATE"; then
    echo "install: downloading template ${OSTEMPLATE}"
    pveam update
    pveam download "$TEMPLATE_STORAGE" "$OSTEMPLATE"
  fi

  pct create "$CTID" "$TEMPLATE_PATH" \
    --hostname "$CT_HOSTNAME" \
    --cores "$CORES" \
    --memory "$MEMORY" \
    --swap 512 \
    --rootfs "${STORAGE}:${DISK_GB}" \
    --net0 "name=eth0,bridge=${BRIDGE},ip=${CT_IP}/${CT_CIDR},gw=${CT_GW}" \
    --unprivileged "$UNPRIVILEGED" \
    --features "$FEATURES" \
    --onboot "$START_ON_BOOT" \
    --start 1

  sleep 5
fi

pct start "$CTID" >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
  if pct exec "$CTID" -- true >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "install: fetching private cluster .env into CT"
CREDS_TMP=$(mktemp)
if [[ -z "${GITEA_TOKEN:-}" ]]; then
  echo "install: GITEA_TOKEN required to fetch private rayify_real_infra_creds" >&2
  echo "  write /root/lexis-markets.secrets.env with: GITEA_TOKEN=..." >&2
  exit 1
fi
curl -fsSL -H "Authorization: token ${GITEA_TOKEN}" "$CREDS_REPO_RAW" -o "$CREDS_TMP"
sed -i 's/\r$//' "$CREDS_TMP"
if ! grep -q '^MARKETS_SUPERVISOR_STATE=' "$CREDS_TMP"; then
  echo "MARKETS_SUPERVISOR_STATE=${STATE_DIR}/state.db" >>"$CREDS_TMP"
fi
pct exec "$CTID" -- bash -lc "mkdir -p $(dirname "$ENV_FILE") $(dirname /opt/.env)"
pct push "$CTID" "$CREDS_TMP" "$ENV_FILE"
rm -f "$CREDS_TMP"

echo "install: syncing app tree into CT ${APP_ROOT}"
pct exec "$CTID" -- bash -lc "rm -rf ${APP_ROOT} && mkdir -p ${APP_ROOT}"
tar -C "$WORK_DIR" -cf - \
  --exclude .git \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.supervisor' \
  . | pct exec "$CTID" -- tar -C "$APP_ROOT" -xf -

echo "install: bootstrap inside CT (Python/uv/systemd — no npm)"
pct exec "$CTID" -- env \
  APP_ROOT="$APP_ROOT" \
  APP_USER="$APP_USER" \
  STATE_DIR="$STATE_DIR" \
  ENV_FILE="$ENV_FILE" \
  REPO_ROOT="$APP_ROOT" \
  bash "${APP_ROOT}/deploy/bootstrap-ct.sh"

if [[ -n "${HA_GROUP:-}" ]]; then
  echo "install: adding to HA"
  ha-manager add "ct:${CTID}" || true
fi

echo
echo "Done."
echo "  CT ${CTID} (${CT_HOSTNAME}) @ ${CT_IP}"
echo "  runtime: Python + systemd (not npm / not Docker-in-LXC)"
echo "  logs: pct exec ${CTID} -- journalctl -u lexis-markets -f"
echo "  status: pct exec ${CTID} -- systemctl status lexis-markets"
echo "  enter: pct enter ${CTID}"
