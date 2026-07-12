# Gaze Redirection — Train & Eval
# Usage:
#   make train-64   # train 64x64 on GPU 0
#   make eval-64    # eval  64x64 on GPU 0
#   ...

VGG  := ./vgg_16.ckpt
DATA := ./dataset/all/
PY   := .conda/bin/python

# ---------------------------------------------------------------------------
# 64x64  (GPU 0)
# ---------------------------------------------------------------------------
train-64:
	CUDA_VISIBLE_DEVICES=0 DATA_ROOT=data/columbia/64x64 \
	$(PY) main.py \
		--mode train \
		--data_path $(DATA) \
		--log_dir ./log/64x64 \
		--image_size 64 \
		--batch_size 64 \
		--vgg_path $(VGG)

eval-64:
	CUDA_VISIBLE_DEVICES=0 DATA_ROOT=data/columbia/64x64 \
	$(PY) main.py \
		--mode eval \
		--data_path $(DATA) \
		--log_dir ./log/64x64 \
		--image_size 64 \
		--batch_size 64 \
		--vgg_path $(VGG)

# ---------------------------------------------------------------------------
# 128x128  (GPU 1)
# ---------------------------------------------------------------------------
train-128:
	CUDA_VISIBLE_DEVICES=1 DATA_ROOT=data/columbia/128x128 \
	$(PY) main.py \
		--mode train \
		--data_path $(DATA) \
		--log_dir ./log/128x128 \
		--image_size 128 \
		--batch_size 64 \
		--vgg_path $(VGG)

eval-128:
	CUDA_VISIBLE_DEVICES=1 DATA_ROOT=data/columbia/128x128 \
	$(PY) main.py \
		--mode eval \
		--data_path $(DATA) \
		--log_dir ./log/128x128 \
		--image_size 128 \
		--batch_size 64 \
		--vgg_path $(VGG)

# ---------------------------------------------------------------------------
# 256x256  (GPU 2)
# ---------------------------------------------------------------------------
train-256:
	CUDA_VISIBLE_DEVICES=2 DATA_ROOT=data/columbia/256x256 \
	$(PY) main.py \
		--mode train \
		--data_path $(DATA) \
		--log_dir ./log/256x256 \
		--image_size 256 \
		--batch_size 64 \
		--vgg_path $(VGG)

eval-256:
	CUDA_VISIBLE_DEVICES=2 DATA_ROOT=data/columbia/256x256 \
	$(PY) main.py \
		--mode eval \
		--data_path $(DATA) \
		--log_dir ./log/256x256 \
		--image_size 256 \
		--batch_size 64 \
		--vgg_path $(VGG)

# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
.PHONY: train-64 eval-64 train-128 eval-128 train-256 eval-256
