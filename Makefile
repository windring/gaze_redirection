# Gaze Redirection - Train & Eval

VGG := ./vgg_16.ckpt
PY  := .conda/bin/python

# ---------------------------------------------------------------------------
# Columbia
# ---------------------------------------------------------------------------
train-columbia-64:
	CUDA_VISIBLE_DEVICES=2 $(PY) main.py \
		--mode train \
		--dataset columbia \
		--data_path /dev/shm/columbia/64x64 \
		--log_dir ./log/columbia_64x64 \
		--image_size 64 \
		--batch_size 256 \
		--vgg_path $(VGG)

eval-columbia-64:
	CUDA_VISIBLE_DEVICES=0 $(PY) main.py \
		--mode eval \
		--dataset columbia \
		--data_path /dev/shm/columbia/64x64 \
		--log_dir ./log/columbia_64x64 \
		--image_size 64 \
		--batch_size 256 \
		--vgg_path $(VGG)

train-columbia-128:
	CUDA_VISIBLE_DEVICES=4 $(PY) main.py \
		--mode train \
		--dataset columbia \
		--data_path /dev/shm/columbia/128x128 \
		--log_dir ./log/columbia_128x128 \
		--image_size 128 \
		--batch_size 128 \
		--vgg_path $(VGG)

eval-columbia-128:
	CUDA_VISIBLE_DEVICES=1 $(PY) main.py \
		--mode eval \
		--dataset columbia \
		--data_path /dev/shm/columbia/128x128 \
		--log_dir ./log/columbia_128x128 \
		--image_size 128 \
		--batch_size 64 \
		--vgg_path $(VGG)

train-columbia-256:
	CUDA_VISIBLE_DEVICES=2 $(PY) main.py \
		--mode train \
		--dataset columbia \
		--data_path /dev/shm/columbia/256x256 \
		--log_dir ./log/columbia_256x256 \
		--image_size 256 \
		--batch_size 32 \
		--vgg_path $(VGG)

eval-columbia-256:
	CUDA_VISIBLE_DEVICES=2 $(PY) main.py \
		--mode eval \
		--dataset columbia \
		--data_path /dev/shm/columbia/256x256 \
		--log_dir ./log/columbia_256x256 \
		--image_size 256 \
		--batch_size 32 \
		--vgg_path $(VGG)

# ---------------------------------------------------------------------------
# XGaze
# ---------------------------------------------------------------------------
train-xgaze-64:
	CUDA_VISIBLE_DEVICES=2 $(PY) main.py \
		--mode train \
		--dataset xgaze \
		--data_path /dev/shm/xgaze/64x64 \
		--log_dir ./log/xgaze_64x64 \
		--image_size 64 \
		--batch_size 256 \
		--min_lightness 0.3 \
		--vgg_path $(VGG)

eval-xgaze-64:
	CUDA_VISIBLE_DEVICES=3 $(PY) main.py \
		--mode eval \
		--dataset xgaze \
		--data_path /dev/shm/xgaze/64x64 \
		--log_dir ./log/xgaze_64x64 \
		--image_size 64 \
		--batch_size 64 \
		--min_lightness 0.3 \
		--vgg_path $(VGG)

train-xgaze-128:
	CUDA_VISIBLE_DEVICES=5 $(PY) main.py \
		--mode train \
		--dataset xgaze \
		--data_path /dev/shm/xgaze/128x128 \
		--log_dir ./log/xgaze_128x128 \
		--image_size 128 \
		--batch_size 128 \
		--min_lightness 0.3 \
		--vgg_path $(VGG)

eval-xgaze-128:
	CUDA_VISIBLE_DEVICES=4 $(PY) main.py \
		--mode eval \
		--dataset xgaze \
		--data_path /dev/shm/xgaze/128x128 \
		--log_dir ./log/xgaze_128x128 \
		--image_size 128 \
		--batch_size 64 \
		--min_lightness 0.3 \
		--vgg_path $(VGG)

.PHONY: \
	train-columbia-64 eval-columbia-64 \
	train-columbia-128 eval-columbia-128 \
	train-columbia-256 eval-columbia-256 \
	train-xgaze-64 eval-xgaze-64 \
	train-xgaze-128 eval-xgaze-128
