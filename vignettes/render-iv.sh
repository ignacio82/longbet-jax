#!/usr/bin/env bash
set -euo pipefail

# Use /home/ignacio/book's prebuilt image, with both LongBet interfaces pinned
# to the IV development snapshot. Override these variables for another checkout.
chapter_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
engine_repo=${LONGBET_REPO:-/home/ignacio/longbet-jax}
engine_ref=${LONGBET_REF:-802cad26ac86f22ba59cd11860feffa556dfd896}
image_name=${BOOK_IMAGE:-book}
image_id=$(docker image inspect --format '{{.Id}}' "$image_name")
engine_commit=$(git -C "$engine_repo" rev-parse --verify "${engine_ref}^{commit}")
render_tmp=$(mktemp -d "${TMPDIR:-$chapter_dir}/longbet-iv-render.XXXXXXXX")
trap 'rm -rf -- "$render_tmp"' EXIT
mkdir "$render_tmp/engine"
git -C "$engine_repo" archive "$engine_commit" \
  R src inst man DESCRIPTION NAMESPACE pyproject.toml README.md |
  tar -x -C "$render_tmp/engine"

docker run --rm \
  -v "$chapter_dir:/chapter" \
  -v "$render_tmp/engine:/engine:ro" \
  -w /chapter \
  -e PYTHONPATH=/engine/src \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e RETICULATE_PYTHON=/opt/longbet-venv/bin/python \
  -e R_LIBS_USER=/tmp/iv-r-library \
  -e LONGBET_R_LIBRARY=/tmp/iv-r-library \
  -e LONGBET_ENGINE_REF="$engine_commit" \
  -e LONGBET_RENDER_IMAGE="$image_id" \
  -e OMP_NUM_THREADS=2 \
  -e OPENBLAS_NUM_THREADS=2 \
  "$image_id" bash -c '
    set -euo pipefail
    mkdir -p "$R_LIBS_USER"
    R CMD INSTALL --library="$R_LIBS_USER" /engine
    quarto render iv.qmd "$@"
  ' render-iv "$@"
