# Update package lists and install Python and pip
sudo apt update
sudo apt install python3 python3-pip

# Install PyTorch with CUDA support (if you have a CUDA-enabled GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install the Hugging Face transformers library
pip install transformers

# Install NVIDIA drivers if using GPU
sudo apt install nvidia-driver-<version>
