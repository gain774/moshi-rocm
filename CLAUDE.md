# moshi-rocm

moshi.cpp の ROCm/HIP 移植版。

## 技術スタック
- C++20, CMake 3.14+
- ggml (HIP バックエンド: `-DGGML_HIP=ON`)
- SentencePiece, FFmpeg 7+, SDL2

## 構成
- `src/` — コアライブラリ（GPU処理はggml抽象レイヤー経由、CUDA直接コードなし）
- `tools/` — CLI デモツール群 (moshi-tts, moshi-stt, moshi-sts 等)
- `cmake/FindGGML.cmake` — ggml検出（HIP含む）
- `scripts/build-rocm.sh` — ROCmビルドスクリプト
- `include/moshi/` — 公開API

## ビルドコマンド
```bash
# ggml (HIP)
cmake .. -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON

# moshi-rocm
cmake .. -DGGML_INCLUDE_DIR=<ggml>/include -DGGML_LIBRARY_DIR=<ggml>/build-hip/src

# 一括ビルド
./scripts/build-rocm.sh --gpu-targets gfx1100
```

## 重要ルール
- ROCm 6.x 系を使用（7.x は hipBLAS API 変更により非推奨）
- GPU処理の変更はggml側で行う（moshi.cpp側にCUDAコードなし）
- フォーク元: Codes4Fun/moshi.cpp、upstream として管理
