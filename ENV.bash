# Basic configuration
export RUN_LOCAL="True"  # Set to "True" for local development

# Tokenizer configuration
export LOCAL_TOKENIZER_PATH="/root/autodl-fs/model/Llama-3.2-11B-Vision-Instruct/tokenizer.json"  # Path to local tokenizer
export LOCAL_MODEL_PATH="/root/autodl-fs/model/Llama-3.2-11B-Vision-Instruct"  # Path to local tokenizer

# Search API configurations
export LOCAL_TEXT_MODEL_PATH="/root/autodl-fs/model/all-MiniLM-L6-v2"  # Path to local text model
export LOCAL_IMAGE_MODEL_PATH="/root/autodl-fs/model/clip-vit-large-patch14-336"  # Path to local image model
export LOCAL_WEB_DATASET_ID="/root/autodl-tmp/dataset/web-search-index-validation"   # ID for local web dataset
export LOCAL_IMAGE_DATASET_ID="/root/autodl-tmp/dataset/image-search-index-validation" # ID for local image dataset

export LOCAL_TEXT_MODEL_PATH="/root/autodl-tmp/small-models/all-MiniLM-L6-v2"  # Path to local text model
export LOCAL_IMAGE_MODEL_PATH="/root/autodl-tmp/small-models/clip-vit-large-patch14-336"  # Path to local image model

# OpenAI configuration for semantic evaluation
export LOCAL_OPENAI_API_KEY="sk-902a9392f76643fab2283fc2c6b6bc34"   # API key for local OpenAI instance
export LOCAL_OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"  # Base URL for local OpenAI instance
export LOCAL_EVAL_MODEL_NAME="qwen-max"

#dataset
export SINGLE_TURN_DATASET_PATH="/root/autodl-tmp/dataset/crag-mm-single-turn-public"
export MULTI_TURN_DATASET_PATH="/root/autodl-tmp/dataset/crag-mm-multi-turn-public"
export EVAL_DATASET_PATH="$SINGLE_TURN_DATASET_PATH"
