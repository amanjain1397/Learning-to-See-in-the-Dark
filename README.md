# Learning-to-See-in-the-Dark
Pytorch implementation of Learning to See in the Dark (CVPR 2018)

Developed a pipeline for processing low-light images, based on end-to-end training of a fully convolutional network. The network operates directly on raw sensor data and the original paper referenced can be found [here](https://arxiv.org/abs/1805.01934).

### Data
For this implementation, only images from the Sony camera has been used. The author has provided this dataset [here](https://storage.googleapis.com/isl-datasets/SID/Sony.zip). The resolution is **4240x2832** for each raw image. I process these images and convert each of them into .npy file for reducing the training time.

### Usage
Producing image
```bash
python main.py eval --dark_image </path/to/raw/image> --model_path </path/to/saved/model> --output_image </path/to/output/image> --cuda 1
```
* `--dark_image` path of the raw image (ARW) you want to evaluate
* `--model` saved model to be used
* `--output_image` path for saving the output image
* `--cuda` set it to 1 for running on GPU, 0 for CPU. (default is 1)

Train model
```bash
python main.py train --train_txt_file </path/to/train/text/file> --patch_size 512 --epochs 4000 --save_model_dir </path/to/save-model/folder> --cuda 1
```
* `--train_txt_file` path to the txt file corresponding to the training data, default is ./Sony/Sony_train_list.txt
* `--patch_size` patch size used while training, default is 512
* `--epochs` number of training epochs, default is 4000
* `--save_model_dir` path to folder where trained model will be saved, default is ./checkpoint_result/
* `--cuda` set it to 1 for running on GPU, 0 for CPU. (default is 1)

Refer to ./main.py for other command line arguments.

### More Info
Learn about U-Net architecture [here](https://arxiv.org/abs/1505.04597). Know about [Rawpy](https://pypi.org/project/rawpy/).
