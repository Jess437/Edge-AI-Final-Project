## 備註
因為要用到nvcc，在T4上建議使用miniconda安裝cuda-toolkit
微調與量化模型是在其他機器上做的

## 1.安裝並啟動miniconda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
conda create --name sg python=3.10
conda activate sg
```

## 2.安裝套件
```bash
chmod +x install.sh
./install.sh
```

## 3.執行程式
```bash
python result.py
```