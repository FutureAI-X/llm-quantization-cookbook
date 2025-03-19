# llm-quantization-cookbook
大语言模型量化

## 一 环境说明
以下为作者基础环境说明
- Python版本：3.12.3
- 操作系统：WSL Ubuntu 24.04.2 LTS

## 二 依赖清单
### 1.1 requirements.txt
项目根路径下的 requirements.txt 文件详细记录了项目的依赖包版本信息。
### 1.2 pip 命令清单
也提供了 pip 命令清单，用于安装依赖包。详见项目根路径下的 pip_command.txt

## 三 量化方法

## 3.1 bitsandbytes
依赖安装
```commandline
pip install transformers accelerate bitsandbytes>0.37.0
```