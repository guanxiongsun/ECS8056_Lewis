# env.sh: paths and environment for running the probe on Isambard-AI (project b5cs).
#   source scripts/isambard/env.sh
export SGVLA_ROOT=/projects/b5cs/SGVLA
export SGVLA_DATA=$SGVLA_ROOT/v2
export HF_HOME=$SGVLA_ROOT/hf
export UV_CACHE_DIR=${SCRATCHDIR}/uv-cache
# The venv's activate script reads unset variables, so relax `set -u` around it.
case $- in *u*) _restore_u=1; set +u ;; *) _restore_u=0 ;; esac
source "$SGVLA_ROOT/venv/bin/activate"
[ "$_restore_u" = 1 ] && set -u
unset _restore_u
