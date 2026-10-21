#!/usr/bin/env bash
# Installs the libraries the trainer needs on top of PyTorch.
# Tested: Python 3.13, PyTorch 2.11.0+cu130 (Colab's build; keep it), CUDA 13.0 driver,
# NVIDIA A100-SXM4-80GB. Needs Linux and an NVIDIA GPU with BF16 (Ampere or newer).
set -euo pipefail
python -c "import torch; assert torch.cuda.is_available(), 'CUDA PyTorch required'; print('torch', torch.__version__)"
pip install -q "flash-linear-attention==0.5.2" "fla-core==0.5.2" einops tokenizers datasets numpy
pip install -q "git+https://github.com/microsoft/dion.git@58d38ad"
python -c "import fla, dion; print('fla', fla.__version__, '| dion ok')"
echo "Next: python exp.py check   then   python exp.py prepare   then   python exp.py train"
