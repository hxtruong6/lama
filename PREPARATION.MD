<!-- conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y -->
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

poetry use system && poetry install

pip install -r requirements_cus.txt
```
1. Install library in requirements_2.txt (for custom package with the previous enviroments)
2. Prepare pretrained model (download and put in the folder)
