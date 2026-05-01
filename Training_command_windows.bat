@echo off
REM =============================================================================
REM MUSUBI TUNER - QWEN IMAGE 2512 TRAINING (WINDOWS CMD VERSION)
REM Replace all [REPLACE_*] placeholders with your actual paths/values
REM =============================================================================

accelerate launch ^
  --num_cpu_threads_per_process 1 ^
  --mixed_precision bf16 ^
  src/musubi_tuner/qwen_image_train_network.py ^
  ^
  REM === MODEL PATHS === ^
  --dit [REPLACE_DIT_MODEL_PATH] ^
  --vae [REPLACE_VAE_MODEL_PATH] ^
  --text_encoder [REPLACE_TEXT_ENCODER_PATH] ^
  ^
  REM === DATASET === ^
  --dataset_config [REPLACE_DATASET_CONFIG_TOML] ^
  --train_batch_size 1 ^
  --gradient_accumulation_steps 1 ^
  --max_data_loader_n_workers 2 ^
  --persistent_data_loader_workers ^
  --enable_bucket ^
  --bucket_no_upscale ^
  --caption_extension .txt ^
  --cache_directory [REPLACE_CACHE_DIR] ^
  ^
  REM === RESOLUTION === ^
  --resolution 1328 1328 ^
  ^
  REM === TRAINING === ^
  --max_train_epochs 2 ^
  --max_train_steps 90000 ^
  --learning_rate 5e-6 ^
  --lr_scheduler constant ^
  --lr_warmup_steps 0 ^
  --optimizer_type AdaFactor ^
  --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False" "weight_decay=0.01" ^
  --max_grad_norm 0 ^
  --gradient_checkpointing ^
  --seed 99 ^
  ^
  REM === PRECISION & MEMORY === ^
  --mixed_precision bf16 ^
  --full_bf16 ^
  --dit_dtype bfloat16 ^
  --vae_dtype bfloat16 ^
  --text_encoder_dtype bfloat16 ^
  --blocks_to_swap 33 ^
  --use_pinned_memory_for_block_swap ^
  --sdpa ^
  --split_attn ^
  --fused_backward_pass ^
  ^
  REM === FLOW MATCHING === ^
  --timestep_sampling sigmoid ^
  --discrete_flow_shift 2.2 ^
  --weighting_scheme mode ^
  --mode_scale 1 ^
  --guidance_scale 4 ^
  --logit_mean 0 ^
  --logit_std 1 ^
  --sigmoid_scale 1 ^
  ^
  REM === NETWORK (LoRA) === ^
  --network_module [REPLACE_NETWORK_MODULE] ^
  --network_dim [REPLACE_NETWORK_DIM] ^
  --network_alpha [REPLACE_NETWORK_ALPHA] ^
  ^
  REM === OUTPUT === ^
  --output_dir [REPLACE_OUTPUT_DIR] ^
  --output_name [REPLACE_MODEL_NAME] ^
  --logging_dir [REPLACE_LOGS_DIR] ^
  --save_every_n_epochs 1 ^
  --save_last_n_epochs 2 ^
  --save_state ^
  --save_state_on_train_end ^
  --mem_eff_save ^
  ^
  REM === RESUME (optional) === ^
  --resume [REPLACE_RESUME_PATH] ^
  ^
  REM === SAMPLING DURING TRAINING (optional) === ^
  --sample_every_n_epochs 1 ^
  --sample_prompts [REPLACE_SAMPLE_PROMPTS_FILE] ^
  --sample_width 1328 ^
  --sample_height 1328 ^
  --sample_steps 20 ^
  --sample_cfg_scale 4 ^
  --sample_seed 99 ^
  --sample_negative_prompt "" ^
  --sample_output_dir [REPLACE_SAMPLE_OUTPUT_DIR] ^
  ^
  REM === MISC === ^
  --disable_prompt_enhancement ^
  --create_missing_captions ^
  --log_prefix [REPLACE_LOG_PREFIX] ^
  --training_comment "[REPLACE_TRAINING_COMMENT]"