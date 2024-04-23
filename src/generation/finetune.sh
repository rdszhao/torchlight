export MODEL_NAME='stabilityai/stable-diffusion-2'
export OUTPUT_DIR='model'
export DATASET_NAME='../../data/finetune_imgs'

accelerate launch train_text_to_image_lora.py \
	--pretrained_model_name_or_path=$MODEL_NAME \
	--train_data_dir=$DATASET_NAME \
	--dataloader_num_workers=0 \
	--resolution=512 --center_crop --random_flip \
	--train_batch_size=1 \
	--gradient_accumulation_steps=4 \
	--max_train_steps=12000 \
	--learning_rate=1e-04 \
	--max_grad_norm=1 \
	--lr_scheduler='cosine' --lr_warmup_steps=0 \
	--output_dir=${OUTPUT_DIR} \
	--checkpointing_steps=1500 \
	--seed=1337