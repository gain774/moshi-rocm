#!/usr/bin/env bash
#
# moshi-rocm build script
# Builds ggml with HIP backend and then builds moshi-rocm against it.
#
# Usage:
#   ./scripts/build-rocm.sh [OPTIONS]
#
# Options:
#   --gpu-targets <targets>   GPU architectures (default: gfx1100)
#   --rocm-path <path>        ROCm install path (default: /opt/rocm)
#   --build-type <type>       CMake build type (default: Release)
#   --jobs <n>                Parallel jobs (default: $(nproc))
#   --skip-ggml               Skip ggml build (use existing)
#   --help                    Show this help
#
# Examples:
#   ./scripts/build-rocm.sh --gpu-targets gfx1030
#   ./scripts/build-rocm.sh --gpu-targets "gfx1100;gfx1030"

set -euo pipefail

# ── Defaults ──
GPU_TARGETS="${GPU_TARGETS:-gfx1100}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
SKIP_GGML=false

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GGML_DIR="$PROJECT_DIR/../ggml"
GGML_BUILD_DIR="$GGML_DIR/build-hip"

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-targets)  GPU_TARGETS="$2"; shift 2 ;;
        --rocm-path)    ROCM_PATH="$2"; shift 2 ;;
        --build-type)   BUILD_TYPE="$2"; shift 2 ;;
        --jobs)         JOBS="$2"; shift 2 ;;
        --skip-ggml)    SKIP_GGML=true; shift ;;
        --help)
            head -20 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validate environment ──
echo "=== ROCm Environment Check ==="

if [ ! -d "$ROCM_PATH" ]; then
    echo "ERROR: ROCm not found at $ROCM_PATH"
    echo "Install ROCm 6.x: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    exit 1
fi

if ! command -v hipcc &>/dev/null; then
    echo "WARNING: hipcc not in PATH. Adding $ROCM_PATH/bin"
    export PATH="$ROCM_PATH/bin:$PATH"
fi

echo "ROCm path:    $ROCM_PATH"
echo "GPU targets:  $GPU_TARGETS"
echo "Build type:   $BUILD_TYPE"
echo "Parallel jobs: $JOBS"
echo ""

# ── Step 1: Build ggml with HIP ──
if [ "$SKIP_GGML" = false ]; then
    echo "=== Step 1: Building ggml with HIP backend ==="

    if [ ! -d "$GGML_DIR" ]; then
        echo "Cloning ggml..."
        git clone https://github.com/ggml-org/ggml "$GGML_DIR"
    fi

    mkdir -p "$GGML_BUILD_DIR"
    cd "$GGML_BUILD_DIR"

    cmake "$GGML_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
        -DGGML_HIP=ON \
        -DGPU_TARGETS="$GPU_TARGETS" \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON

    cmake --build . --parallel "$JOBS"

    echo ""
    echo "ggml HIP build complete: $GGML_BUILD_DIR"
    echo ""
else
    echo "=== Step 1: Skipping ggml build (--skip-ggml) ==="
    echo ""
fi

# ── Step 2: Build moshi-rocm ──
echo "=== Step 2: Building moshi-rocm ==="

MOSHI_BUILD_DIR="$PROJECT_DIR/build"
mkdir -p "$MOSHI_BUILD_DIR"
cd "$MOSHI_BUILD_DIR"

# Detect SentencePiece (common install locations)
SP_ARGS=""
if [ -d "/usr/local/include/sentencepiece" ]; then
    SP_ARGS="-DSentencePiece_INCLUDE_DIR=/usr/local/include -DSentencePiece_LIBRARY_DIR=/usr/local/lib"
elif [ -d "$HOME/repos/sentencepiece/src" ]; then
    SP_ARGS="-DSentencePiece_INCLUDE_DIR=$HOME/repos/sentencepiece/include -DSentencePiece_LIBRARY_DIR=$HOME/repos/sentencepiece/lib"
fi

# Detect FFmpeg
FF_ARGS=""
if [ -d "$HOME/lib/ffmpeg-master-latest-linux64-lgpl-shared" ]; then
    FF_ARGS="-DFFmpeg_DIR=$HOME/lib/ffmpeg-master-latest-linux64-lgpl-shared"
fi

cmake "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGGML_INCLUDE_DIR="$GGML_DIR/include" \
    -DGGML_LIBRARY_DIR="$GGML_BUILD_DIR/src" \
    $SP_ARGS \
    $FF_ARGS

cmake --build . --parallel "$JOBS"

echo ""
echo "=== Build Complete ==="
echo "Binaries: $MOSHI_BUILD_DIR/bin/"
echo ""
echo "Next steps:"
echo "  1. Copy ggml libraries to bin/:"
echo "     cp $GGML_BUILD_DIR/src/libggml*.so $MOSHI_BUILD_DIR/bin/"
echo "     cp $GGML_BUILD_DIR/src/ggml-hip/libggml-hip.so $MOSHI_BUILD_DIR/bin/"
echo "  2. Download models:  aria2c --disable-ipv6 -i tools/moshi-defaults.txt"
echo "  3. List devices:     ./bin/moshi-tts -l"
echo "  4. Run TTS:          ./bin/moshi-tts -d HIP0 \"Hello World!\""
