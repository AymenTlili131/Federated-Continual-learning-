# To get setup : 
all the following files should go Under "New" after you clone this repo 
MSE.ipynb and wsiayn are the notebooks that handle :

+ load Autoencoder architecture "from Double_input_transformer import CustomDataset,TransformerAE"
+ Loading data from pre-defined .npy lists 
+ training & wandb logging 
+ Auto-encoder testset forward passes 
+ finetuning 
+ Some cleaning tasks on testset logs when KL divergence and WS distance isn't calculated or logged properly they had to be recomputed from the weights directly
+ finetuning on limited size data (20%,10%,5%)(epoch-wise,step-wise precision)
+ create the boxplots for post finetuning analysis on the test set for various loss implementation


the file TDA of weights implements the Mapper algotihm for each loss functions "AE epoch {0/best} {num_epochs_training}" and an indication of the loss function sometimes in the name .
that file splits the Auto-encoder testset forward passes into Ground Truth (GT) , Freshly predicted weights (PD) and Finetuned Weights (FN)

so the notebooks and .npy should be on the same folder
a seperate "./data/" should contains the "merged_zoo.csv" file that contains all of the model weights and their performance ,as well as what activation function and when (epoch number) they were sampled
the "./data/" should also have a folder of SplitMNIST , train /test and each class within is in a folder holding it's name .
from our 40 epoch training loop that has an early stopping condition triggered after 10th epoch if the model performance on tha validation set stagnate within a certain margin , we manage to create a varied zoo but due to compute reasons we limit the training on "gelu" activation and the 10th epoch where all class combinations are logged and early stopping counter just started .
This guarantees high performance models

the model zoo was trained using files such as Silu.py and target.py . feel free to inspect them 
