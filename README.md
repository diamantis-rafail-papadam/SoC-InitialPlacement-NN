# Technical University of Crete - Thesis
This work is part of my thesis for the completion of my studies in Technical University of Crete.

You can download the ".soclog" files as well as a pretrained CNN from this [google drive](https://drive.google.com/drive/folders/1y3_COABOSd1N7PeNLltROEFipIch7THJ).
- For the full datasets, of size 500,000 each, you shall download zip files from the "LARGE" folder.
- For a smaller version of the datasets, of size 100,000 each, you shall download zip files from the "SMALL" folder.

> Note that "DEFAULT_DROID4_500000.zip" did not fit in the 15GB of free google drive space.  
> Therefore "DEFAULT_DROID4_100000.zip" was added, in the "LARGE" folder as well.  
> It shall be uploaded proprely once the availability of more space in the cloud is secured.

Steps to execute the code:
1. Download [miniconda](https://docs.conda.io/projects/miniconda/en/latest)
2. conda create --name <specify_conda_environment_name> python=3.8
3. Run `git clone https://github.com/diamantis-rafail-papadam/SoC-InitialPlacement-NN.git` and `cd SoC-InitialPlacement-NN`.
4. Run `pip install -r requirements.txt`.
> You might need to download another pytorch version according to your NVIDIA CUDA Driver.  
> If this is the case, check [current versions](https://pytorch.org/get-started/locally/) and [old versions](https://pytorch.org/get-started/previous-versions/).
5. Go inside the created folder and unzip the downloaded dataset there.
> You should see a "logs" folder created which contains all ".soclog" files.
> If you want to see the win ratio of each player, run "python count_wins.py" (inside the created folder where "logs" are located).  
> The first number you see is the win ratio for each player while the second number is the average score over all ".soclog" files.
6. Make a directory named "DATASET", this will be needed in the next step.
7. Run "python extract_log_data.py" which might take a few minutes.
8. Run "python preprocess_data.py" which will create the input for the neural network.
- Regarding the "produce_graphs.py" file, you might want to change a few things:
  - Line 16 &nbsp; | Feel free to change the hyperparameters.
  - Line 125 | The path for the basic pretrained model, if you have one.
  - Line 126 | The path for the cnn pretrained model, if you have one.
  - Line 180 | This is the number of epochs.
  - Line 184 | Choose a device for the basic model, according to your system availability.
  - Line 187 | Choose a device for the cnn model, according to your system availability.
9. Run "python produce_graphs.py" and enjoy the results!

> As a final note, you can use the "train.py" script to pre-train either network in whatever way you like.
