#!/usr/bin/env bash
# Bootstrap Lexis Markets supervisor inside the LXC. Invoked by install.sh via pct exec.
set -euo pipefail

APP_ROOT="${APP_ROOT:-/opt/lexis-markets}"
APP_USER="${APP_USER:-lexis}"
STATE_DIR="${STATE_DIR:-/var/lib/lexis-markets}"
ENV_FILE="${ENV_FILE:-/etc/lexis-markets.env}"
REPO_ROOT="${REPO_ROOT:-/opt/lexis-markets-deploy}"
# Must match Ray cluster (currently 3.12.x on k3s workers)
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

export DEBIAN_FRONTEND=noninteractive
export LANG=C.UTF-8

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git \
  build-essential libpq-dev libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
  liblzma-dev tk-dev uuid-dev

if ! id -u "$APP_USER" >/dev/null 2>&1; then
  useradd --system --home-dir "$STATE_DIR" --create-home --shell /usr/sbin/nologin "$APP_USER"
fi

mkdir -p "$APP_ROOT" "$STATE_DIR" "$(dirname "$ENV_FILE")"
# dotenv ROOT = parents[2] of config.py → /opt when package lives at /opt/lexis-markets/lexis_markets
ln -sfn "$ENV_FILE" /opt/.env

if [[ ! -d "$APP_ROOT/lexis_markets" ]]; then
  echo "bootstrap: missing $APP_ROOT/lexis_markets" >&2
  exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "bootstrap: missing $ENV_FILE (creds not installed)" >&2
  exit 1
fi

# Install uv → managed CPython matching the Ray cluster (Debian 12 is 3.11 by default).
# Keep the interpreter outside /root so the lexis service user can exec it.
export PATH="/root/.local/bin:/usr/local/bin:${PATH}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/opt/uv-python}"
mkdir -p "$UV_PYTHON_INSTALL_DIR"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if [[ -x /root/.local/bin/uv ]]; then
  ln -sfn /root/.local/bin/uv /usr/local/bin/uv
fi
if [[ -x /root/.cargo/bin/uv ]]; then
  ln -sfn /root/.cargo/bin/uv /usr/local/bin/uv
fi
hash -r || true
command -v uv >/dev/null 2>&1 || { echo "bootstrap: uv not on PATH" >&2; ls -la /root/.local/bin /usr/local/bin/uv 2>&1 || true; exit 1; }

uv python install "$PYTHON_VERSION"
chmod -R a+rX "$UV_PYTHON_INSTALL_DIR"
rm -rf "$APP_ROOT/.venv"
uv venv --python "$PYTHON_VERSION" "$APP_ROOT/.venv"
# shellcheck disable=SC1091
source "$APP_ROOT/.venv/bin/activate"
uv pip install --python "$APP_ROOT/.venv/bin/python" -r "$APP_ROOT/requirements.txt"

PYVER=$("$APP_ROOT/.venv/bin/python" -c 'import sys; print("%d.%d"%sys.version_info[:2])')
echo "bootstrap: venv python=${PYVER}"
if [[ "$PYVER" != "$PYTHON_VERSION" ]]; then
  echo "bootstrap: expected Python ${PYTHON_VERSION}, got ${PYVER}" >&2
  exit 1
fi

install -m 0644 "$REPO_ROOT/deploy/lexis-markets.service" /etc/systemd/system/lexis-markets.service
sed -i "s#WorkingDirectory=.*#WorkingDirectory=${APP_ROOT}#" /etc/systemd/system/lexis-markets.service
sed -i "s#EnvironmentFile=.*#EnvironmentFile=${ENV_FILE}#" /etc/systemd/system/lexis-markets.service
sed -i "s#Environment=PYTHONPATH=.*#Environment=PYTHONPATH=${APP_ROOT}#" /etc/systemd/system/lexis-markets.service
sed -i "s#Environment=MARKETS_SUPERVISOR_STATE=.*#Environment=MARKETS_SUPERVISOR_STATE=${STATE_DIR}/state.db#" /etc/systemd/system/lexis-markets.service
sed -i "s#ExecStart=.*#ExecStart=${APP_ROOT}/.venv/bin/python -m lexis_markets.supervisor.main#" /etc/systemd/system/lexis-markets.service
sed -i "s#^User=.*#User=${APP_USER}#" /etc/systemd/system/lexis-markets.service
sed -i "s#^Group=.*#Group=${APP_USER}#" /etc/systemd/system/lexis-markets.service

chown -R "$APP_USER:$APP_USER" "$APP_ROOT" "$STATE_DIR"
chmod 640 "$ENV_FILE"
chown root:"$APP_USER" "$ENV_FILE"

systemctl daemon-reload
systemctl enable lexis-markets.service
systemctl restart lexis-markets.service

echo "bootstrap: lexis-markets.service started"
systemctl --no-pager --full status lexis-markets.service || true
