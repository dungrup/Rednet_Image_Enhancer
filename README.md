# RED-Net

This repository is implementation of the "Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections". <br />
To reduce computational cost, it adopts stride 2 for the first convolution layer and the last transposed convolution layer.

Credits: https://github.com/yjn870/REDNet-pytorch

## Requirements
- PyTorch
- tqdm
- Numpy
- Pillow

## Results

<table>
    <tr>
        <td><center>Input</center></td>
        <td><center>JPEG (Quality 10)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/monarch.bmp" height="300"></center>
    	</td>
    	<td>
    		<center><img src="./data/monarch_jpeg_q10.png" height="300"></center>
    	</td>
    </tr>
    <tr>
        <td><center>AR-CNN</center></td>
        <td><center><b>RED-Net 10</b></center></td>
    </tr>
    <tr>
        <td>
        	<center><img src="./data/monarch_ARCNN.png" height="300"></center>
        </td>
        <td>
        	<center><img src="./data/monarch_REDNet10.png" height="300"></center>
        </td>
    </tr>
    <tr>
        <td><center><b>RED-Net 20</b></center></td>
        <td><center><b>RED-Net 30</b></center></td>
    </tr>
    <tr>
        <td>
        	<center><img src="./data/monarch_REDNet20.png" height="300"></center>
        </td>
        <td>
        	<center><img src="./data/monarch_REDNet30.png" height="300"></center>
        </td>
    </tr>
</table>

## Usages

### Train

- In [dataset.py](dataset.py) vary the train/val set sizes. By default, I am using first 50 tfrecords for training and the next 20 (50-70) as validation set. (Keep it as is now)

- [dataset.py](dataset.py) also currently takes 5Mbps images for Waymo (in line #15) and 1Mbps images for BDD Dataset (line #128)

- Expected data structure for raw and comp_images_dir is:

```
raw_images_dir
  |
   ------- tfrecord_xxxx
        |
         ------- *.png or *.jpg


comp_images_dir 
    |
     --------- tfrecord_xxxx
        |
         ------ val_cbr_xMbps_xMbuf
            |
             ---------- val 
                |
                 -------- *.png
```

- When training begins, the model weights will be saved every epoch. <br />
If you want to train quickly, you should use **--use_fast_loader** option.

```bash
python train.py --arch "REDNet30" \  # REDNet10, REDNet20, REDNet30               
               --raw_images_dir "" \
               --comp_images_dir "" \
               --outputs_dir "" \  # Where the weights are stored
               --patch_size 50 \
               --batch_size 2 \
               --num_epochs 20 \
               --lr 1e-4 \
               --threads 8 \
               --seed 123            
```

### Test

Output results consist of image compressed with JPEG and image with artifacts reduced.

```bash
python example --arch "REDNet30" \  # REDNet10, REDNet20, REDNet30
               --weights_path "" \
               --image_path "" \
               --outputs_dir "" \
               --jpeg_quality 10               
```
