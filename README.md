# Rule Network with Selective Logical Operators(RNS)
## Requirements
torch>=1.8.0  
torchvision>=0.9.0  
tensorboard>=1.15.0  
sklearn>=0.23.2  
numpy>=1.19.2  
pandas>=1.1.3  
matplotlib>=3.3.2  
CUDA>=11.1  

## Run the demo
We need to put the data sets in the dataset folder. You can specify one data set in the dataset folder and train the model as follows:
    ```
   python3 -W ignore experiment.py -d bank  -bs 32 -s 15@32@32 -e 100 -lrde 100 -lr 0.001 -ki 0 -mp 12441 -i 0 -wd 1e-8 -bin 15
    ```

