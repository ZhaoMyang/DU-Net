### Environment
Pytorch 1.7.1, Python 3.6


### Style Transfer with Single-image
To train the model and obtain the image, run

```
python train_CLIPstyler.py --content_path test/1.jpg --content_name test --exp_name exp1 --text "Render the product photo in a retro 8-bit pixel art style, with dithered shading and a CRT screen glow effect"
```
