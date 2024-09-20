# To get setup : 
all the following files should go Under "New" after you clone this repo 
MSE.ipynb and wsiayn are the notebooks that handle :

+ load Autoencoder architecture
  > "from Double_input_transformer import CustomDataset,TransformerAE"
+ Loading data from pre-defined .npy lists 
+ training & wandb logging 
+ Auto-encoder testset forward passes 
+ finetuning 
+ Some cleaning tasks on testset logs when KL divergence and WS distance isn't calculated or logged properly they had to be recomputed from the weights directly
+ finetuning on limited size data (20%,10%,5%)(epoch-wise,step-wise precision)
+ create the boxplots for post finetuning analysis on the test set for various loss implementation


*the file TDA of weights implements the **Mapper** algotihm for each loss functions "AE epoch {0/best} {num_epochs_training}.csv" and an indication of the loss function sometimes in the name .
*that file splits the Auto-encoder testset forward passes into Ground Truth (GT) , Freshly predicted weights (PD) and Finetuned Weights (FN)

-so the notebooks and .npy should be on the same folder
-a seperate "./data/" should contains the "merged_zoo.csv" file that contains all of the model weights and their performance ,as well as what activation function and when (epoch number) they were sampled
-the "./data/" should also have a folder of SplitMNIST , train /test and each class within is in a folder holding it's name as well as DistilledSplitMNIST variant .
-from our 40 epoch training loop that has an early stopping condition triggered after 10th epoch if the model performance on tha validation set stagnate within a certain margin , we manage to create a varied zoo but due to compute reasons we limit the training on "gelu" activation and the 10th epoch where all class combinations are logged and early stopping counter just started .
-This guarantees high performance models where it's meaningful to combine their knowledge

the model zoo was trained using files such as Silu.py and target.py . feel free to inspect them 

+ Zoo.csv and Distilled data and loadable npy : https://drive.google.com/drive/folders/1_dOrA9PlHdLpoJHyeWwACRKcF-RLTigp?usp=sharing
+ Screenshots used during the making of this proposal : https://drive.google.com/drive/folders/1MSocnYbi-nzVE5GVbNPO-ont64scrvEL?usp=drive_link
+ Prediction and Finetuning  weights as well as some metrics and distances : https://drive.google.com/drive/folders/16sbcyhWOfVtU1HGpFoOls0RrXxTk4TI0?usp=sharing

this repo provides better methods https://github.com/VICO-UoE/DatasetCondensation on advanced datasets than the base dataset distillation method we opted to use because it was already available for MNIST https://github.com/SsnL/dataset-distillation
