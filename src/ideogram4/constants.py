SEQUENCE_PADDING_INDICATOR = -1

OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3
# Clean (un-noised) latents of a source/reference image, fed as in-context tokens
# for image-editing. Distinguished from the target via a separate MRoPE frame index
# (see REFERENCE_POSITION_T in train_edit.py). Unused at plain text-to-image inference.
REFERENCE_IMAGE_INDICATOR = 4

# Image grid coordinates start at this offset so they never collide with text token indices
# (text positions start at 0 and never exceed max_text_tokens, which is well below this).
IMAGE_POSITION_OFFSET = 65536

# Layers of Qwen3-VL whose hidden states are concatenated and fed to the transformer.
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
