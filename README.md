# Technical University of Crete - Thesis
This work is part of my thesis for the completion of my studies in Technical University of Crete.

You can download the ".soclog" files from this [google drive](https://drive.google.com/drive/folders/1y3_COABOSd1N7PeNLltROEFipIch7THJ).
- For the full datasets, of size 500,000 each, you shall download zip files from the "LARGE" folder.
- For a smaller version of the datasets, of size 100,000 each, you shall download zip files from the "SMALL" folder.

> Note that "DEFAULT_DROID4_500000.zip" did not fit in the 15GB of free google drive space.  
> Therefore "DEFAULT_DROID4_100000.zip" was added, even in the "LARGE" folder.  
> It shall be uploaded proprely once the availability of more space in the cloud is secured.

Steps to execute the code:
1. Create a folder for the zipped dataset that you're interested in.
2. Go inside the folder you've created and unzip the dataset there.
> You should see a "logs" folder created which contains all ".soclog" files.  
> If you want to see the win ratio of each player, run "python count_wins.py" (inside the created folder where "logs" are located).  
> The first number you see is the win ratio for each player while the second number is the average score over all ".soclog" files.
3. Make a directory named "DATASET", this will be needed in the next step.
4. Run "python extract_log_data.py" which might take a few minutes.
5. Run "python preprocess_data.py" which will create the input for the neural network.
- Regarding the "produce_graphs.py" file, you might want to change a few things:
  - Line 16 &nbsp; | Feel free to change the hyperparameters.
  - Line 125 | The path for the basic pretrained model, if you have one.
  - Line 126 | The path for the cnn pretrained model, if you have one.
  - Line 180 | This is the number of epochs.
  - Line 184 | Choose a graphics card for basic model, or even run the network on the 'cpu' based on your system availability.
  - Line 187 | Choose a graphics card for the cnn, or even run the network on the 'cpu' based on your system availability.
6. Run "python produce_graphs.py" and enjoy the results!
