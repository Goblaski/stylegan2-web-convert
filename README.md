# stylegan2-web-convert
A converter and some examples to run official [StyleGAN2](https://github.com/NVlabs/stylegan2) based networks in your browser using ONNX. This approach may work in the future for [StyleGAN3](https://github.com/NVlabs/stylegan3) as NVLabs stated on their StyleGAN3 git: "This repository is an updated version of stylegan2-ada-pytorch". However, StyleGAN3 current uses ops not supported by ONNX (affine_grid_generator).

**I am not affilated with NVLabs.** I am merely created this repository to help others with converting their StyleGAN2 models for use in a browser.
- For the narrative/blogpost see: https://www.guidodejong.nl/hack/running-stylegan2-ada-in-browser/
- For a working example see: https://www.guidodejong.nl/research/experimental/art-generator-v2/

# Instructions (Short version of the blogpost)
## Running StyleGAN2-ada in browser
StyleGAN2(-ada) is known for generating photorealistic images of e.g. portraits and is now widely used in research, education and entertainment. Deployment of these models can be either by providing the files directly, dedicated model hosting services or in-browser. The latter has so far been the most challenging, yet a good trade-off between usability and cost. Since my last try a couple of months ago, I’ve finally seen new technologies to properly achieve this. So in this post I will explain the steps required to run StyleGAN2-ada models in your browser using [onnxruntime](https://github.com/microsoft/onnxruntime) with the current knowledge and technologies (October 2021). Tested on Windows 10.

## Requirements
- [StyleGAN2(-ada) PyTorch edition](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [PyTorch 1.9.1](https://pytorch.org/get-started/locally/) (this differs from the StyleGAN2-ada recommendations)
- A StyleGAN2 model (either torch or tensorflow based)
  - For tensorflow based models you need to convert them with the [legacy converter](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/legacy.py)

## Preparation
It is recommended that you have your models in SyleGAN2-ada PyTorch format. Either you have trained your model previously in StyleGAN2(-ada) PyTorch or you can convert the tensorflow versions using the [legacy converter](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/legacy.py). This should allow you to use any official StyleGAN2 version models. Example code can be found below.
```
python legacy.py \
    --source=https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkl \
    --dest=stylegan2-cat-config-f.pkl
```

## PyTorch model to ONNX
For the next step we need to convert your model to the [ONNX](https://onnx.ai/) format. I’ve written a script here to load the model, assign all the input variables and have Y as the final output variable. Three onnx files will be created; a full model, the mapping model and the synthesis model which can be used in combination with the HTML files. It uses opset 10 of ONNX. The code will probably give you some warnings, but in the end you will end up with a file with the onnx extension. This configuration was the only configuration that worked properly for me.
```
python .\onnx_convert.py \
    --source=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl
    --dest=/path/of/your/choice/with/the/html/files/
```

## Minimalist html + js
Two minimalist html files with included javascript can be found in the repository. The index.html will show you plain StyleGAN2 image generation. The index_transfer.html will show a basic styletransfer example. Place these files in the same folder as your onnx files. This might not work locally, so I suggest trying this on a web server. Do note that for this example I am using a fixed 512\*512 resolution for ease. The official [StyleGAN2-ada PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch) has some examples with this resolution. In case your model uses a different resolution, change the image resolution values!

## Demo
https://www.guidodejong.nl/research/experimental/art-generator-v2/
