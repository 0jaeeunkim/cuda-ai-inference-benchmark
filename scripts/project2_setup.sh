#!/usr/bin/env bash
# Run these stages manually in the server terminal. Never install into base.
set -euo pipefail

PROJECT2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT2_CONDA="${PROJECT2_CONDA:-$HOME/miniforge3/bin/conda}"
PROJECT2_CACHE="$PROJECT2_ROOT/.cache/huggingface"
PROJECT2_ENV=cuda-project2
PROJECT2_RELEASE=26.9.4-0
cd "$PROJECT2_ROOT"

download() {
  local project2_url="$1" project2_destination="$2"
  local project2_partial="$project2_destination.partial"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 "$project2_url" -o "$project2_partial" || return
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --timeout=30 -O "$project2_partial" "$project2_url" || return
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$project2_url" "$project2_partial" <<'PY' || return
import shutil
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=60) as response:
    with open(sys.argv[2], "wb") as destination:
        shutil.copyfileobj(response, destination)
PY
  else
    printf '%s\n' 'A downloader is required: curl, wget, or python3.' >&2
    return 1
  fi
  mv -- "$project2_partial" "$project2_destination"
}

require_conda() {
  if [[ ! -x "$PROJECT2_CONDA" ]]; then
    printf 'Conda is missing at %s. Run: bash scripts/project2_setup.sh install-conda\n' "$PROJECT2_CONDA" >&2
    return 1
  fi
}

require_env() {
  require_conda
  if ! "$PROJECT2_CONDA" run --no-capture-output -n "$PROJECT2_ENV" python --version >/dev/null 2>&1; then
    printf 'Environment %s is unavailable. Run: bash scripts/project2_setup.sh create-env\n' "$PROJECT2_ENV" >&2
    return 1
  fi
}

case "${1:---help}" in
  --help|-h)
    printf '%s\n' 'Stages: all | install-conda | create-env | install-packages | download-cache'
    ;;
  all)
    # Each child must succeed before the next stage starts.
    for project2_stage in install-conda create-env install-packages download-cache; do
      bash "$PROJECT2_ROOT/scripts/project2_setup.sh" "$project2_stage"
    done
    ;;
  install-conda)
    if [[ -x "$PROJECT2_CONDA" ]]; then
      "$PROJECT2_CONDA" --version
      exit 0
    fi
    if [[ -e "$HOME/miniforge3" ]]; then
      printf '%s\n' 'Existing ~/miniforge3 is incomplete; inspect it before retrying.' >&2
      exit 1
    fi
    PROJECT2_INSTALLERS="$PROJECT2_ROOT/.cache/project2-installers"
    mkdir -p "$PROJECT2_INSTALLERS"
    cd "$PROJECT2_INSTALLERS"
    PROJECT2_INSTALLER="Miniforge3-$PROJECT2_RELEASE-Linux-x86_64.sh"
    PROJECT2_INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/download/$PROJECT2_RELEASE/$PROJECT2_INSTALLER"
    download "$PROJECT2_INSTALLER_URL" "$PROJECT2_INSTALLER"
    download "$PROJECT2_INSTALLER_URL.sha256" "$PROJECT2_INSTALLER.sha256"
    sha256sum --check "$PROJECT2_INSTALLER.sha256"
    bash "$PROJECT2_INSTALLER" -b -p "$HOME/miniforge3"
    "$PROJECT2_CONDA" --version
    ;;
  create-env)
    require_conda
    project2_environments="$("$PROJECT2_CONDA" env list --json)"
    project2_environment_present="$(python3 -c '
import json, pathlib, sys
envs = json.load(sys.stdin)["envs"]
print("yes" if any(pathlib.Path(path).name == sys.argv[1] for path in envs) else "no")
' "$PROJECT2_ENV" <<< "$project2_environments")"
    if [[ "$project2_environment_present" == yes ]]; then
      require_env
      if ! "$PROJECT2_CONDA" run --no-capture-output -n "$PROJECT2_ENV" python -c \
        'import sys; raise SystemExit(sys.version_info[:3] != (3, 10, 20))'; then
        printf '%s\n' 'Existing cuda-project2 has a different Python version; inspect it before continuing.' >&2
        exit 1
      fi
      printf '%s\n' 'Reusing existing cuda-project2 environment.'
    else
      "$PROJECT2_CONDA" create --name "$PROJECT2_ENV" --override-channels --channel conda-forge \
        python=3.10.20 pip --yes
    fi
    "$PROJECT2_CONDA" run --no-capture-output -n "$PROJECT2_ENV" python --version
    ;;
  install-packages)
    require_env
    "$PROJECT2_CONDA" run --no-capture-output -n "$PROJECT2_ENV" python -m pip install \
      --only-binary=:all: --index-url https://download.pytorch.org/whl/cu130 'torch==2.12.0+cu130'
    "$PROJECT2_CONDA" run --no-capture-output -n "$PROJECT2_ENV" python -m pip install \
      --only-binary=:all: --index-url https://pypi.org/simple -r docs/project2/requirements-hf.txt
    "$PROJECT2_CONDA" run --no-capture-output -n "$PROJECT2_ENV" python -m pip check
    mkdir -p results/project2/setup
    "$PROJECT2_CONDA" run --no-capture-output -n "$PROJECT2_ENV" python -m pip freeze \
      > results/project2/setup/pip-freeze.txt
    "$PROJECT2_CONDA" list -n "$PROJECT2_ENV" --explicit > results/project2/setup/conda-explicit.txt
    ;;
  download-cache)
    require_env
    export HF_HOME="$PROJECT2_CACHE"
    export HF_HUB_CACHE="$PROJECT2_CACHE/hub"
    export HF_DATASETS_CACHE="$PROJECT2_CACHE/datasets"
    "$PROJECT2_CONDA" run --no-capture-output -n "$PROJECT2_ENV" \
      python src/project2_pipeline.py --mode prepare
    ;;
  *)
    printf 'Unknown stage: %s\n' "$1" >&2
    exit 2
    ;;
esac
