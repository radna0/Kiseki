export DEBIAN_FRONTEND=noninteractive

curl -f https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo apt-key add -


sudo apt-get update -y
sudo apt-get install software-properties-common -y

DEBIAN_FRONTEND=noninteractive sudo add-apt-repository ppa:deadsnakes/ppa -y


sudo apt install python3.10 python3.10-venv python3.10-dev -y
python3.10 --version

sudo curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.10

sudo pip uninstall -y tensorflow tensorflow-cpu


sudo apt-get install -y libgl1 libglib2.0-0 google-perftools



# Memory check
wget https://raw.githubusercontent.com/radna0/EasyAnimate/refs/heads/TPU/mem_check.py

# tpu-info
python3.10 -m pip install tpu-info


# Pytorch XLA
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip install torch_scatter


rm -rf ckpt/basicpbc.pth
wget -O ckpt/basicpbc.pth https://huggingface.co/radna/Kiseki-ckpt/resolve/main/basicpbc.pth

