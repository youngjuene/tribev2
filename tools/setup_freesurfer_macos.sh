#!/usr/bin/env bash
set -euo pipefail

# Download and install FreeSurfer for macOS arm64.
#
# This script follows the public FreeSurfer 8.2.0 macOS release path.  It needs
# administrator privileges for the pkg installer and a user-provided FreeSurfer
# license file for the tools to run after installation.

VERSION="${FREESURFER_VERSION:-8.2.0}"
ARCH="$(uname -m)"
if [[ "${ARCH}" != "arm64" ]]; then
  echo "This helper currently targets Apple Silicon arm64 Macs; got ${ARCH}." >&2
  exit 2
fi

PKG_NAME="freesurfer-macOS-darwin_arm64-${VERSION}.pkg"
PKG_URL="https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/${VERSION}/${PKG_NAME}"
PKG_MD5="${FREESURFER_PKG_MD5:-64ac912e2ec53f811f133b59d3271cf1}"
DOWNLOAD_DIR="${FREESURFER_DOWNLOAD_DIR:-/private/tmp}"
PKG_PATH="${DOWNLOAD_DIR}/${PKG_NAME}"
INSTALL_ROOT="${FREESURFER_INSTALL_ROOT:-/Applications/freesurfer}"
FREESURFER_HOME="${FREESURFER_HOME:-${INSTALL_ROOT}/${VERSION}}"
FS_LICENSE="${FS_LICENSE:-${HOME}/license.txt}"

mkdir -p "${DOWNLOAD_DIR}"

echo "FreeSurfer version: ${VERSION}"
echo "Package URL: ${PKG_URL}"
echo "Package path: ${PKG_PATH}"
echo "FREESURFER_HOME: ${FREESURFER_HOME}"
echo "FS_LICENSE: ${FS_LICENSE}"

if [[ ! -f "${PKG_PATH}" ]]; then
  curl -L -C - -o "${PKG_PATH}" "${PKG_URL}"
fi

if command -v md5 >/dev/null 2>&1; then
  actual_md5="$(md5 -q "${PKG_PATH}")"
  if [[ "${actual_md5}" != "${PKG_MD5}" ]]; then
    echo "MD5 mismatch for ${PKG_PATH}: expected ${PKG_MD5}, got ${actual_md5}" >&2
    exit 3
  fi
fi

if [[ "${1:-}" == "--download-only" ]]; then
  echo "Download verified. Skipping installation."
  exit 0
fi

sudo installer -pkg "${PKG_PATH}" -target /

if [[ ! -f "${FS_LICENSE}" ]]; then
  cat >&2 <<EOF
FreeSurfer installed, but no license file was found at:
  ${FS_LICENSE}

Set FS_LICENSE to your FreeSurfer license.txt path before running the tools.
EOF
  exit 4
fi

export FREESURFER_HOME
export FS_LICENSE
export NO_FSFAST="${NO_FSFAST:-1}"
# shellcheck disable=SC1091
source "${FREESURFER_HOME}/SetUpFreeSurfer.sh"

command -v mri_surf2vol
command -v mri_surfcluster
command -v mri_surf2surf
echo "FreeSurfer setup complete."
