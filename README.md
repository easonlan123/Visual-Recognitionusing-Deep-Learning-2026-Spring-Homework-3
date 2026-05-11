# NYCU Computer Vision 2026 HW3

* **Student ID:** 111550017
* **Name:** 藍逸薰

## Introduction

 The core idea is to build a Mask R-CNN detector-segmentor with pretrained weight ResNet-50 backbone by layer freezing and resizing and then carefully control post-processing at inference time.
 Model: https://drive.google.com/file/d/1w_QfCugn4HFSpCLbYKX5iP7ZcYCf2g0W/view?usp=sharing

---

## Environment Setup

How to install dependencies.

```bash
pip install -r requirements.txt
```
---

## Usage

# Training

folder format:  
folder 

&emsp; model.py 

&emsp; predict.py  
&emsp; utils.py  

&emsp; hw3-data-release 

&emsp;&emsp; train  
&emsp;&emsp;&emsp; images  
&emsp;&emsp; test  
&emsp;&emsp;&emsp; images  
&emsp;&emsp; val  
&emsp;&emsp;&emsp; images 



# Training
```bash
python model.py
```
# Testing

```bash
python predict.py
```
---

## Performance Snapshot

<img width="1182" height="52" alt="Image" src="leaderboard.png" />
