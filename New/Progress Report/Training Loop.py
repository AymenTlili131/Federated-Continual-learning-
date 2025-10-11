

           
track=0

from torchinfo import summary

warnings.filterwarnings("ignore")


global track , output , Brain ,model
grads=None


accelerator = Accelerator(mixed_precision="bf16")
nb_scen=len(os.listdir("./data/Scenario/"))
for t in range(nb_scen):
    namef=os.listdir("./data/Scenario/")[t]
    train_pair2 = np.load("./data/Scenario/"+os.listdir("./data/Scenario/")[t]+"/train pair.npy", allow_pickle=True)
    print("./data/Scenario/"+os.listdir("./data/Scenario/")[t]+"/"+"train pair.npy" ,len(train_pair2))
    val_pair2 = np.load("./data/Scenario/"+os.listdir("./data/Scenario/")[t]+"/val pair.npy", allow_pickle=True)
    test_pair2 = np.load("./data/Scenario/"+os.listdir("./data/Scenario/")[t]+"/test pair.npy", allow_pickle=True)
    
    train_pair2 = [ list(x) for x in train_pair2]
    test_pair2 = [ list(x) for x in test_pair2]
    val_pair2 = [ list(x) for x in val_pair2]
    
    #random.shuffle(test_pair2)
    #random.shuffle(val_pair2)
    
    #if len(train_pair2)>1000 and  len(val_pair2)>1000 and  len(test_pair2)>100:
    if namef=='overlapping 0 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 11 gelu' :
        for combination in all_losses[3:]:
            d_loos=dict()
            DF= pd.DataFrame(columns=Cols)
            track=0


            cnn_samples=5
            num_epochs=6
            batch_size=36
            batch_limit=10000
            epochs_fn=2

            if not(os.path.isdir(f".//Experiments//{os.listdir('.//data//Scenario//')[t]}")):
                os.mkdir(f".//Experiments//{os.listdir('.//data//Scenario//')[t]}")
            results_path=f".//Experiments//{os.listdir('.//data//Scenario//')[t]}//{combination } {str(datetime.datetime.now())} {num_epochs} /"
            if not(os.path.isdir(results_path)):
                os.mkdir(results_path)
            if not(os.path.isdir(results_path+f'Tracking/')):
                os.mkdir(results_path+f'Tracking/')
            if not(os.path.isdir(results_path+f'Attention/')):
                os.mkdir(results_path+f'Attention/')
    
            cnn_acc_ID=[]
            cnn_acc_OOD=[]
            
            Lambda=0
            step_size=0
            factor=0
            threshhold=0
            threshold_mode = 0
            eps=0
        
            
            ch=torch.load('./AE epoch{} {} {}.pth'.format( "initial" , "original", -1),map_location="cpu")
            mod=TransformerAE(max_seq_len=50,
                                    N=4,
                                    heads=4,
                                    d_model=960,#960
                                    d_ff=960,#960
                                    neck=512,#256
                                    dropout=0.07)#0.12)
            mod.load_state_dict(ch['model_state_dict'])
            del(ch)
            minimal_loss=100000
            print(summary(mod))
            lrE1=0.15 #trial.suggest_float("Learning_rate",0.0002,0.5)
            lrE2=0.04 
            lrL=0.15 
            lrD=0.085
            optimizerEnc1 = Adam(mod.enc1.parameters(), lr=lrE1,eps=1e-10,weight_decay=0.005)
            optimizerEnc2 = Adadelta(mod.enc2.parameters(), lr=lrE2,eps=1e-10,weight_decay=0.005)
            optimizerDense = SGD(mod.vec2neck.parameters(), lr=lrL,weight_decay=0.005)
            optimizerDec = Adadelta(mod.dec.parameters(), lr=lrD,eps=1e-10,weight_decay=0.01)
            mod = torch.compile(mod)
    
    
    
            sched_name="CyclicLR"
            if sched_name=="CyclicLR" :
                step_size=24000#trial.suggest_int("step_size_up",900,2800)
                schedulerEnc1 = torch.optim.lr_scheduler.CyclicLR(optimizerEnc1, base_lr=1e-4, max_lr=lrE1, step_size_up=step_size, step_size_down=80000,scale_mode="iterations",mode="triangular2",cycle_momentum=False)
                schedulerEnc2 = torch.optim.lr_scheduler.CyclicLR(optimizerEnc2, base_lr=1e-4, max_lr=lrE2, step_size_up=step_size, step_size_down=80000,scale_mode="iterations",mode="triangular2",cycle_momentum=False)
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizerDense, base_lr=1e-4, max_lr=lrD, step_size_up=8000, step_size_down=8000,scale_mode="iterations",mode="triangular",cycle_momentum=False)

    
            resume_epoch=0
            # ch=torch.load("/media/crns/ADATA HD330/Experiments/mixed model/AE epoch 0 600.pth",map_location="cpu")
            # resume_epoch=ch["epoch"]
            # mod.load_state_dict(ch['model_state_dict'])
            # optimizerEnc1.load_state_dict(ch['optimizerENC1_state_dict'])
            # optimizerEnc2.load_state_dict(ch['optimizerENC2_state_dict'])
            # optimizerDense.load_state_dict(ch['optimizerDense_state_dict'])
            # optimizerDec.load_state_dict(ch['optimizerDec_state_dict'])
    
            #del(ch)
    
            criterion = nn.MSELoss()
            LW=LWLN_loss()
            LossWS=SamplesLoss('sinkhorn')
            LWs=LWLN_loss_single()
    
            device = accelerator.device
            run = wandb.init(
            # Set the project where this run will be logged
            project="WR4CL-project",
            name= f"{namef} {num_epochs} {combination} {str(datetime.datetime.now())}" ,
            # Track hyperparameters and run metadata
            config={
                "Scenario":namef,
                "instances in train":len(train_pair2),
                "Loss":combination,
                "lr Encoder 1": lrE1,
                "lr Encoder 2": lrE2,
                "lr Linear": lrL,
                "lr Decoder": lrD,
                "epochs": num_epochs,
                "sched_name": sched_name,
                "step_size": step_size,
                "batch_size": batch_size,
                "step_size":step_size,
                "Lambda_contractive":Lambda
            },)
    
            #run = wandb.init(project="aymen-project", id="ar1497fk", resume="must")
    
            # "factor":factor,
            #     "threshhold":threshhold,
            #     "threshold_mode":threshold_mode,
            #     "eps":eps,
            #random.shuffle(train_pair2)
            cs_tr=CustomDataset(train_pair2,batch_size=batch_size,batch_limit=batch_limit)
            cs_tr=accelerator.prepare(cs_tr)
            
            nb_batches = len(cs_tr)
            
            print("Number of training batchs:",nb_batches)
            cs_val=CustomDataset(val_pair2,batch_size=batch_size,batch_limit=batch_limit)
            nb_val_batches = len(cs_val)

            cs_ts=CustomDataset(test_pair2,batch_size=batch_size,batch_limit=batch_limit)
    
            mod, optimizerEnc1,optimizerEnc2,optimizerDense,optimizerDec, cs_tr= accelerator.prepare(mod, optimizerEnc1,optimizerEnc2,optimizerDense,optimizerDec,cs_tr)
            # wandb.watch(mod, log_freq=10000 ,criterion=criterion,
            #     log='parameters',
            #     log_graph=True)
            #mod.train()
            
            mod.train()
            
            wandb.define_metric("custom_step")
            wandb.define_metric("Prediction Histogram", step_metric='custom_step')
            wandb.define_metric("Total Histogram", step_metric='custom_step')
            used_Loss=0.0
            for epoch in tqdm(range(resume_epoch,num_epochs),total=num_epochs):
                # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(locals().items())), key= lambda x: -x[1])[:10]:
                #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
                print("entering ",epoch)
                random.shuffle(train_pair2)
                
                #start_time_epoch = time.time()
                for i in range(nb_batches):#nb_batches for training'''
                    start_time = time.time()
                    
                    Dataset,EXP,ACC,U = cs_tr[i]
                    
                    x1,x2,tg = Dataset[:,0,:], Dataset[:,1,:],Dataset[:,2,:]
    
                    x1=x1.cuda() #.to(torch.float32)
                    x2=x2.cuda() #.to(torch.float32)
                    tg=tg.cuda() #.to(torch.float32)
    
                    optimizerEnc1.zero_grad()
                    optimizerEnc2.zero_grad()
                    optimizerDense.zero_grad()
                    optimizerDec.zero_grad()
    
                    #output = mod(x1,x2)
    
                    d_loos,output,mod=update_all_metrics(d_loos,mod,x1,x2,tg,DF)
                    torch.nn.utils.clip_grad_norm_(mod.parameters(), max_norm=2.0)
    
    
                    used_Loss=0
                    #print(used_Loss,d_loos)
                    if isinstance(combination, list):
                        for pair in d_loos.items():
                            if pair[0]==combination[0] or pair[0]==combination[1] :
                                used_Loss=used_Loss+pair[1]
                    if isinstance(combination, str):
                        for pair in d_loos.items():
                            if pair[0]==combination: 
                                used_Loss=used_Loss+pair[1]
                    #print(combination ,"\t \t",used_Loss)
    
                    accelerator.backward(used_Loss)
                    #loss_tr.backward()
                    frob = frobnorm(mod)
                    optimizerEnc1.step()
                    optimizerEnc2.step()
                    optimizerDense.step()
                    optimizerDec.step()
                    if sched_name=="ReduceLROnPlateau" :
                        scheduler.step(used_Loss)
                    else:
                        scheduler.step()
                        schedulerEnc1.step()
                        schedulerEnc2.step()
                    
                    loss_to_save = float(used_Loss.detach().cpu().item())
                    wandb.log({f"Tracked Loss {combination}":loss_to_save})
                
                
                
                if used_Loss.detach().cpu().item()<minimal_loss:
                    minimal_loss=used_Loss.detach().cpu().item()
                    torch.save({'epoch':epoch,'model_state_dict': mod.state_dict(),
                                'optimizerENC1_state_dict':  optimizerEnc1.state_dict() ,
                                'optimizerENC2_state_dict':optimizerEnc2.state_dict(),
                                'optimizerDense_state_dict':optimizerDense.state_dict(),
                                'optimizerDec_state_dict': optimizerDec.state_dict(),
                                'Batch Loss':used_Loss.detach().cpu().item()},
                                results_path+'AE epoch best_.pth')
    
                end_time = time.time() 
                execution_time = end_time - start_time  
                if execution_time>4000 :
                    break
                if (epoch+1)%50==0 or (epoch+1) in [1,5,10,30]:
                    print("validating")
                    for block in [2,3,4]:
                            for head in range(4):
                                plt.figure(figsize=(20, 20))
                                hm=sns.heatmap(torch.mean( torch.mean(output[block][head], dim=1), dim=0).detach().cpu(), annot=False, cmap='cubehelix')
                                plt.title('Attention Heatmap')
                                heatmap_path = f'heatmap {block-2}_{head}_epoch{epoch}_step_{i}.png'
                                plt.savefig(results_path+"Attention/"+heatmap_path,format='svg', dpi=800)
                                wandb.log({f"attention_heatmap {block-2}_{head}":  wandb.Image(hm,caption=f"attention_heatmap attention_heatmap {block-2}_{head}")})
                                plt.close()
                    mod.eval()
                    loss_val = []
                    for i_val in range(nb_val_batches):#nb_val_batches batches for validation'''
                        #start_time_batch = time.time()
                        Dataset_val,EXP_val,ACC_val,U_val = cs_val[i_val]
                        x1_val,x2_val,tg_val = Dataset_val[:,0,:], Dataset_val[:,1,:],Dataset_val[:,2,:]
                        x1_val=x1_val.cuda() #.to(torch.float32)
                        x2_val=x2_val.cuda() #.to(torch.float32)
                        tg_val=tg_val.cuda() #.to(torch.float32)
                        with torch.no_grad():
                            #output_val = mod(x1_val,x2_val)
                            
                            loss_val.append(criterion(mod(x1_val,x2_val)[0],tg_val))
                            #print(x1_val.shape,mod(x1_val,x2_val)[0].shape)
                    loss_value = sum(loss_val)/len(loss_val)
                    wandb.log({f"val_loss":loss_value})
                    print("entering test set")
                    DF= pd.DataFrame(columns=Cols)
                    track=0
                    nb_test_batches = len(cs_ts)
                    
                    mod,cs_ts= accelerator.prepare(mod, cs_ts)
                    mod.eval()
    
                    #loss_values = []
                    
                    nb_test= min(60,nb_test_batches)
                    print("test batches :",nb_test_batches,"batch size",batch_size,"temporarly testing on batchs:",nb_test)
                    for j in range(nb_test): #nb_test_batches number of batches for testing'''
                        Dataset,EXP,ACC,U = cs_ts[j]
                        x1,x2,tg = Dataset[:,0,:], Dataset[:,1,:],Dataset[:,2,:]
    
                        x1=x1.cuda() #.to(torch.float32)
                        x2=x2.cuda() #.to(torch.float32)
                        tg=tg.cuda() #.to(torch.float32)
                        #with torch.no_grad():
                        #    output = mod(x1,x2)
                        #    loss_values.append(criterion(output[0],tg))
    
                        for vec in range(len(x1)):
                            DF.at[track,"label task 1"]=f'{EXP[vec][0]}'
                            DF.at[track,"label task 2"]=f'{EXP[vec][1]}'
                            #print(Cols[2:2466][-2:],Cols[2466:4930][-2:],Cols[4930:7394][-2:])
    
                            vec1=mod(x1,x2)[0][vec].detach().cpu() #prediction
                            vec2=tg[vec].detach().cpu()            #target
                            DF.loc[track,Cols[2:2466]]=vec2.tolist()
                            DF.loc[track,Cols[2466:4930]]=vec1.tolist()#Cols[4930:7394][-2:])
    
    
    
                            MSE=criterion(vec1,vec2).item()
                            MS1=criterion(vec1[:208],vec2[:208]).item()
                            MS2=criterion(vec1[208:1414],vec2[208:1414]).item()
                            MS3=criterion(vec1[1414:1514],vec2[1414:1514]).item()
                            MS4=criterion(vec1[1514:2254],vec2[1514:2254]).item()
                            MS5=criterion(vec1[2254:],vec2[2254:]).item()
                            #print("MSE :",MS1,MS2,MS3,MS4,MS5,MSE)
                            DF.at[track,"MSE"]=MSE
                            DF.at[track,"MSE 1"]=MS1
                            DF.at[track,"MSE 2"]=MS2
                            DF.at[track,"MSE 3"]=MS3
                            DF.at[track,"MSE 4"]=MS4
                            DF.at[track,"MSE 5"]=MS5
    
    
    
                            kl_loss = nn.KLDivLoss(reduction="mean")
                            KLD=kl_loss(vec1,vec2).item() #pred, true
                            KL1=kl_loss(vec1[:208],vec2[:208]).item()
                            KL2=kl_loss(vec1[208:1414],vec2[208:1414]).item()
                            KL3=kl_loss(vec1[1414:1514],vec2[1414:1514]).item()
                            KL4=kl_loss(vec1[1514:2254],vec2[1514:2254]).item()
                            KL5=kl_loss(vec1[2254:],vec2[2254:]).item()
                            #print("KLD :",KLD,KL1,KL2,KL3,KL4,KL5)
                            DF.at[track,"KLD"]=KLD
                            DF.at[track,"KL 1"]=KL1
                            DF.at[track,"KL 2"]=KL2
                            DF.at[track,"KL 3"]=KL3
                            DF.at[track,"KL 4"]=KL4
                            DF.at[track,"KL 5"]=KL5
    
    
    
    
                            WSD=wasserstein_distance(vec1,vec2)
                            WS1=wasserstein_distance(vec1[:208],vec2[:208])
                            WS2=wasserstein_distance(vec1[208:1414],vec2[208:1414])
                            WS3=wasserstein_distance(vec1[1414:1514],vec2[1414:1514])
                            WS4=wasserstein_distance(vec1[1514:2254],vec2[1514:2254])
                            WS5=wasserstein_distance(vec1[2254:],vec2[2254:])
                            #print("WSD :",WSD,WS1,WS2,WS3,WS4,WS5)
                            DF.at[track,"Wasserstein Loss"]=WSD
                            DF.at[track,"WS 1"]=WS1
                            DF.at[track,"WS 2"]=WS2
                            DF.at[track,"WS 3"]=WS3
                            DF.at[track,"WS 4"]=WS4
                            DF.at[track,"WS 5"]=WS5
    
                            for name, param in mod.named_parameters():
                                if name == 'vec2neck.weight':
                                    W = param
                                    break
    
                            CL=loss_Contractive(mod.vec2neck.weight ,vec1,vec2, output[1], 0.00001)
                            DF.at[track,"contractive distance"]=CL.cpu().item()
    
                            #print("Contractive :",CL.cpu().item())
                            Norm1=np.sum(np.abs(vec1.numpy()))
                            N11=np.sum(np.abs(vec1[:208].numpy()))
                            N12=np.sum(np.abs(vec1[208:1414].numpy()))
                            N13=np.sum(np.abs(vec1[1414:1514].numpy()))
                            N14=np.sum(np.abs(vec1[1514:2254].numpy()))
                            N15=np.sum(np.abs(vec1[2254:].numpy()))
                            #print("Weight pred L1: ",Norm1, N11,N12,N13,N14,N15)
                            DF.at[track,"N1"]=Norm1
                            DF.at[track,"N11"]=N11
                            DF.at[track,"N12"]=N12
                            DF.at[track,"N13"]=N13
                            DF.at[track,"N14"]=N14
                            DF.at[track,"N15"]=N15
    
                            Norm2=np.sum(np.abs(vec2.numpy()))
                            N21=np.sum(np.abs(vec2[:208].numpy()))
                            N22=np.sum(np.abs(vec2[208:1414].numpy()))
                            N23=np.sum(np.abs(vec2[1414:1514].numpy()))
                            N24=np.sum(np.abs(vec2[1514:2254].numpy()))
                            N25=np.sum(np.abs(vec2[2254:].numpy()))
                            #print("Weight GT L1: ",Norm2, N21,N22,N23,N24,N25)
                            DF.at[track,"N2"]=Norm2
                            DF.at[track,"N21"]=N21
                            DF.at[track,"N22"]=N22
                            DF.at[track,"N23"]=N23
                            DF.at[track,"N24"]=N24
                            DF.at[track,"N25"]=N25
    
                            DF.at[track,"saturated in pred(%)"]=100*sum(1 for x in vec1 if x > 0.95 or x<0.05)/len(vec1)
                            DF.at[track,"saturated in GT(%)"]=100*sum(1 for x in vec2 if x > 0.95 or x<0.05)/len(vec2)
                            DF.at[track,"LWLN"]=LWs(vec1,vec2).cpu().item()
                            if vec<cnn_samples and j==0:
                                print(f"Test vector {vec} batch {j} Epoch: {epoch}")
                                figure=get_plot_object(vec1, tg)
                                wandb.log({"Plot of pred vs target":  wandb.Image(figure,caption=f"Test vector {vec} batch {j} Epoch: {epoch} vs pred")})
                                plt.close(figure)
    
                                #import umap
                                # reducer = umap.UMAP()
                                # Batch_UMAP=torch.rand(80,2464)
                                # fitted_reduced = reducer.fit(Batch_UMAP)
                                # fitted_reduced.transform(vec1)
        
                                ###########RECONSTRUCTING##############
                                print("Entering Reconstruction for fine-tuning and eigenvalues")
                                y_pred=torch.unsqueeze(output[0][vec], 0) 
                                y =torch.unsqueeze(tg[vec], 0) 
        
                                selected_row = cs_ts.df.iloc[int(U[vec][0]), 11:17]  
                                columns_with_one = selected_row[selected_row == 1].index.tolist()
                                activ=columns_with_one
                                epochCNN=cs_ts.df.loc[int(U[vec][0])]['epoch']
        
        
                                checkpoint=OrderedDict()
                                checkpoint1=OrderedDict()
                                checkpoint2=OrderedDict()
                                vector_aux= output[0][vec].detach()
                                y_pred=vector_aux.cpu()
        
                                task1=[int(x) for x in EXP[vec][0]]
                                task2=[int(x) for x in EXP[vec][1]]
                                task3=sorted(list(set(task1+task2)))
        
        
                                All=list(range(10))
                                L2=[k for k in All if k not in task3] # Out of distribution classes
                                L_others=[k for k in All if k not in task3] #Classes to test on (In distribution)
        
                                checkpoint["module_list.0.weight"]=torch.tensor(np.array(y_pred[0:200]).reshape([8, 1, 5, 5]))
                                checkpoint["module_list.0.bias"]=torch.tensor(np.array(y_pred[200:208]).reshape([8]))
        
                                checkpoint["module_list.3.weight"]=torch.tensor(np.array(y_pred[208:1408]).reshape([6, 8, 5, 5]))
                                checkpoint["module_list.3.bias"]=torch.tensor(np.array(y_pred[1408:1414]).reshape([6]))
        
                                checkpoint["module_list.6.weight"]=torch.tensor(np.array(y_pred[1414:1510]).reshape([4, 6, 2, 2]))
                                checkpoint["module_list.6.bias"]=torch.tensor(np.array(y_pred[1510:1514]).reshape([4]))
        
                                checkpoint["module_list.9.weight"]=torch.tensor(np.array(y_pred[1514:2234]).reshape([20,36]))
                                checkpoint["module_list.9.bias"]=torch.tensor(np.array(y_pred[2234:2254]).reshape([20]))
        
                                checkpoint["module_list.11.weight"]=torch.tensor(np.array(y_pred[2254:2454]).reshape([10,20]))
                                checkpoint["module_list.11.bias"]=torch.tensor(np.array(y_pred[2454:2464]).reshape([10]))
        
                                Brain = CNN(1,activ[0],0,"kaiming_uniform")
        
                                model=copy.deepcopy(Brain)
                                model.load_state_dict(checkpoint)
                           
                           
                                
                                checkpoint1["module_list.0.weight"]=torch.tensor(np.array(x1[vec][0:200].cpu()).reshape([8, 1, 5, 5]))
                                checkpoint1["module_list.0.bias"]=torch.tensor(np.array(x1[vec][200:208].cpu()).reshape([8]))
        
                                checkpoint1["module_list.3.weight"]=torch.tensor(np.array(x1[vec][208:1408].cpu()).reshape([6, 8, 5, 5]))
                                checkpoint1["module_list.3.bias"]=torch.tensor(np.array(x1[vec][1408:1414].cpu()).reshape([6]))
        
                                checkpoint1["module_list.6.weight"]=torch.tensor(np.array(x1[vec][1414:1510].cpu()).reshape([4, 6, 2, 2]))
                                checkpoint1["module_list.6.bias"]=torch.tensor(np.array(x1[vec][1510:1514].cpu()).reshape([4]))
        
                                checkpoint1["module_list.9.weight"]=torch.tensor(np.array(x1[vec][1514:2234].cpu()).reshape([20,36]))
                                checkpoint1["module_list.9.bias"]=torch.tensor(np.array(x1[vec][2234:2254].cpu()).reshape([20]))
        
                                checkpoint1["module_list.11.weight"]=torch.tensor(np.array(x1[vec][2254:2454].cpu()).reshape([10,20]))
                                checkpoint1["module_list.11.bias"]=torch.tensor(np.array(x1[vec][2454:2464].cpu()).reshape([10]))
                            
                                CNN_x1=copy.deepcopy(Brain)
                                CNN_x1.load_state_dict(checkpoint1)
                                # Extract weights from a specific layer (e.g., 'fc' layer)
                                # for name, param in CNN_x1.named_parameters():
                                #     if len(param.shape)>1:
                                #         weights = param
                                #         weights = weights.view(weights.shape[0], -1)
                                #         weights = weights.T @ weights
                                #         # Compute eigenvalues of the weight matrix
                                #         eigenvalues = torch.linalg.eigvals(weights.detach().cpu()).numpy()
    
                                #         # Extract real and imaginary parts
                                #         real_eigenvalues = eigenvalues.real
                                #         imag_eigenvalues = eigenvalues.imag
    
                                #             # Plot histogram of the real parts of the eigenvalues
                                #         plt.figure(figsize=(8, 6))
                                #         plt.hist(real_eigenvalues, bins=30, alpha=0.7, color='blue', edgecolor='black')
                                #         plt.xlabel('Real Part of Eigenvalues')
                                #         plt.ylabel('Frequency')
                                #         plt.title(f'Input 1 Eigenvalues of layer {name} shaped {tuple(param.shape)} model number {vec}')
                                        
                                #         plt.grid(True)
                                #         #plt.show()
                                #         image_path = results_path+f"Tracking/Input 1 Eigenvalues of layer {name} shaped {tuple(param.shape)} model number {vec}.png"
                                #         plt.savefig(image_path)
                                        
                                #         wandb.log({f"Input 1 Eigenvalues of layer {name} shaped {tuple(param.shape)} model number {vec}":  wandb.Image(image_path,caption=f"Input 1 Eigenvalues of layer {name} shaped {tuple(param.shape)} model number {vec}")})
                                #         plt.close()
                            
    
                                for name, param in CNN_x1.named_parameters():
                                    if len(param.shape) > 1:
                                        weights = param
                                        weights = weights.view(weights.shape[0], -1)
                                        weights = weights.T @ weights
                                        
                                        # Compute eigenvalues of the weight matrix
                                        eigenvalues = torch.linalg.eigvals(weights.detach().cpu()).numpy()
                                        
                                        # Extract real and imaginary parts
                                        real_eigenvalues = eigenvalues.real
                                        imag_eigenvalues = eigenvalues.imag
                                        
                                        # Log histogram to wandb
                                        wandb.log({
                                            f"Input 1 Eigenvalues of layer {name} shaped {tuple(param.shape)}": 
                                            wandb.Histogram(real_eigenvalues, num_bins=64)
                                        })
                                        
                                
                                checkpoint2["module_list.0.weight"]=torch.tensor(np.array(x2[vec][0:200].cpu()).reshape([8, 1, 5, 5]))
                                checkpoint2["module_list.0.bias"]=torch.tensor(np.array(x2[vec][200:208].cpu()).reshape([8]))
        
                                checkpoint2["module_list.3.weight"]=torch.tensor(np.array(x2[vec][208:1408].cpu()).reshape([6, 8, 5, 5]))
                                checkpoint2["module_list.3.bias"]=torch.tensor(np.array(x2[vec][1408:1414].cpu()).reshape([6]))
        
                                checkpoint2["module_list.6.weight"]=torch.tensor(np.array(x2[vec][1414:1510].cpu()).reshape([4, 6, 2, 2]))
                                checkpoint2["module_list.6.bias"]=torch.tensor(np.array(x2[vec][1510:1514].cpu()).reshape([4]))
        
                                checkpoint2["module_list.9.weight"]=torch.tensor(np.array(x2[vec][1514:2234].cpu()).reshape([20,36]))
                                checkpoint2["module_list.9.bias"]=torch.tensor(np.array(x2[vec][2234:2254].cpu()).reshape([20]))
        
                                checkpoint2["module_list.11.weight"]=torch.tensor(np.array(x2[vec][2254:2454].cpu()).reshape([10,20]))
                                checkpoint2["module_list.11.bias"]=torch.tensor(np.array(x2[vec][2454:2464].cpu()).reshape([10]))
                            
                                CNN_x2=copy.deepcopy(Brain)
                                CNN_x2.load_state_dict(checkpoint2)
                                # Extract weights from a specific layer (e.g., 'fc' layer)
                                # for name, param in CNN_x2.named_parameters():
                                #     if len(param.shape)>1:
                                #         weights = param
                                #         weights = weights.view(weights.shape[0], -1)
                                #         weights = weights.T @ weights
                                #         # Compute eigenvalues of the weight matrix
                                #         eigenvalues = torch.linalg.eigvals(weights.detach().cpu()).numpy()
    
                                #         # Extract real and imaginary parts
                                #         real_eigenvalues = eigenvalues.real
                                #         imag_eigenvalues = eigenvalues.imag
    
                                #             # Plot histogram of the real parts of the eigenvalues
                                #         plt.figure(figsize=(8, 6))
                                #         plt.hist(real_eigenvalues, bins=30, alpha=0.7, color='blue', edgecolor='black')
                                #         plt.xlabel('Real Part of Eigenvalues')
                                #         plt.ylabel('Frequency')
                                #         plt.title(f'Input 2 Eigenvalues of layer {name} shaped {tuple(param.shape)} model number {vec}')
                                        
                                #         plt.grid(True)
                                #         #plt.show()
                                #         image_path = results_path+f"Tracking/Input 2 Eigenvalues of layer {name} shaped {tuple(param.shape)} model number {vec}.png"
                                #         plt.savefig(image_path)
                                        
                                #         wandb.log({f"Input 2 Eigenvalues of layer {name} shaped {tuple(param.shape)} model number {vec}":  wandb.Image(image_path,caption=f"Input 2 Eigenvalues of layer {name} shaped {tuple(param.shape)} model number {vec}")})
                                #         plt.close()
                                
    
                                for name, param in CNN_x2.named_parameters():
                                    if len(param.shape) > 1:
                                        weights = param
                                        weights = weights.view(weights.shape[0], -1)
                                        weights = weights.T @ weights
                                        
                                        # Compute eigenvalues of the weight matrix
                                        eigenvalues = torch.linalg.eigvals(weights.detach().cpu()).numpy()
                                        
                                        # Extract real and imaginary parts
                                        real_eigenvalues = eigenvalues.real
                                        imag_eigenvalues = eigenvalues.imag
                                        
                                        # Log histogram to wandb
                                        wandb.log({
                                            f"Input 1 Eigenvalues of layer {name} shaped {tuple(param.shape)}": 
                                            wandb.Histogram(real_eigenvalues, num_bins=64)
                                        })
                                        
                               
                                for name, param in model.named_parameters():
                                    if len(param.shape) > 1:
                                        weights = param
                                        weights = weights.view(weights.shape[0], -1)
                                        weights = weights.T @ weights
                                        
                                        # Compute eigenvalues of the weight matrix
                                        eigenvalues = torch.linalg.eigvals(weights.detach().cpu()).numpy()
                                        
                                        # Extract real and imaginary parts
                                        real_eigenvalues = eigenvalues.real
                                        imag_eigenvalues = eigenvalues.imag
                                        
                                        # Log histogram to wandb
                                        wandb.log({
                                            f"Input 1 Eigenvalues of layer {name} shaped {tuple(param.shape)}": 
                                            wandb.Histogram(real_eigenvalues, num_bins=64)
                                        })
                                        
                                criterion_CNN0=CrossEntropyLoss()
        
                                test_IF0=ClassSpecificImageFolder( root="./data/SplitMnist/test/",dropped_classes=[str(x) for x in L2],transform=transforms.Compose([ transforms.ToTensor(),transforms.Grayscale(1)]))
                                Ts_DL0 = DataLoader(dataset=test_IF0, batch_size=36, num_workers=0, shuffle=False)
        
                                _, valid_epoch_acc0,correct_counts,total_counts= validate(model, Ts_DL0,  criterion_CNN0,10,epoch,vec)
    
                                if len(task3)==10:
                                    valid_epoch_acc1=valid_epoch_acc0
                                    continue
                                else:
                                    criterion_CNN1=CrossEntropyLoss()
                                    test_IF1=ClassSpecificImageFolder( root="./data/SplitMnist/test/",dropped_classes=[str(x) for x in task3],transform=transforms.Compose([ transforms.ToTensor(),transforms.Grayscale(1)]))
                                    Ts_DL1 = DataLoader(dataset=test_IF1, batch_size=36, num_workers=0, shuffle=False)
        
                                valid_epoch_loss0, valid_epoch_acc1,correct_counts,total_counts= validate(model, Ts_DL1,  criterion_CNN1,10,epoch,vec)
                                epoch_nb=-1
    
                                print("Reconstructed cnn acc ID",valid_epoch_acc0)
                                print("Reconstructed cnn acc OOD",valid_epoch_acc1)
                                DF.at[track,"Reconstructed Accuracy ID"]=valid_epoch_acc0
        
                                optimizerCNN = Adam(model.parameters(), lr=0.05)
                                schedulerCNN = torch.optim.lr_scheduler.CyclicLR(optimizerCNN ,base_lr=1e-3, max_lr=0.1, step_size_up=400, mode="triangular2", cycle_momentum=False)
                                criterion_CNN=CrossEntropyLoss()
        
        
                                train_IF0=ClassSpecificImageFolder( root="./data/SplitMnist/train/",dropped_classes=[str(x) for x in L2],transform=transforms.Compose([ transforms.ToTensor(),transforms.Grayscale(1)]))
                                Tr_DLr = DataLoader(dataset=train_IF0, batch_size=36, num_workers=0, shuffle=True)
        
        
                                fine_tune_needed=0
                                #FINETUNING
                                for epoch_cnn in range(epochs_fn):
                                    if epoch_cnn==0:
                                        train_epoch_loss, train_epoch_acc = train(model, Tr_DLr, optimizerCNN, criterion_CNN,10,df=DF,First=True)
                                        valid_epoch_loss0FN, valid_epoch_acc0FN,correct_counts,total_counts= validate(model, Ts_DL0,  criterion_CNN,10,epoch,vec)
                                        DF.at[track,"epoch 0"]=valid_epoch_acc0FN
                                    else:
                                        train_epoch_loss, train_epoch_acc = train(model, Tr_DLr, optimizerCNN, criterion_CNN,10,df=DF)
                                        valid_epoch_loss0FN, valid_epoch_acc0FN,correct_counts,total_counts= validate(model, Ts_DL0,  criterion_CNN,10,epoch,vec)
                                        
                                        DF.at[track,f"epoch {epoch_cnn}"]=valid_epoch_acc0FN
                                    schedulerCNN.step()
                                    fine_tune_needed+=1
        
                                for name, param in model.named_parameters():
                                    if len(param.shape) > 1:
                                        weights = param
                                        weights = weights.view(weights.shape[0], -1)
                                        weights = weights.T @ weights
                                        
                                        # Compute eigenvalues of the weight matrix
                                        eigenvalues = torch.linalg.eigvals(weights.detach().cpu()).numpy()
                                        
                                        # Extract real and imaginary parts
                                        real_eigenvalues = eigenvalues.real
                                        imag_eigenvalues = eigenvalues.imag
                                        
                                        # Log histogram to wandb
                                        wandb.log({
                                            f"Input 1 Eigenvalues of layer {name} shaped {tuple(param.shape)}": 
                                            wandb.Histogram(real_eigenvalues, num_bins=64)
                                        })
                                    
    
                            L_param=[]
                            for param in model.parameters():
                                m = nn.Flatten(0,-1)
                                L_param.append(m(param))
                            vec1FN = torch.Tensor()
                            for idx in L_param:
                                vec1FN = torch.cat((vec1FN, idx.view(-1)))
                            vec1FN=vec1FN.detach().cpu()
                            vec2=tg[vec].cpu()
                            DF.loc[track,Cols[4930:7394]]=vec1FN.tolist()
                            MSE=criterion(vec1FN,vec2).item()
                            MS1=criterion(vec1FN[:208],vec2[:208]).item()
                            MS2=criterion(vec1FN[208:1414],vec2[208:1414]).item()
                            MS3=criterion(vec1FN[1414:1514],vec2[1414:1514]).item()
                            MS4=criterion(vec1FN[1514:2254],vec2[1514:2254]).item()
                            MS5=criterion(vec1FN[2254:],vec2[2254:]).item()
    
    
                            #print("MSE :",MS1,MS2,MS3,MS4,MS5,MSE)
                            DF.at[track,"MSE FN"]=MSE
                            DF.at[track,"MSE 1 FN"]=MS1
                            DF.at[track,"MSE 2 FN"]=MS2
                            DF.at[track,"MSE 3 FN"]=MS3
                            DF.at[track,"MSE 4 FN"]=MS4
                            DF.at[track,"MSE 5 FN"]=MS5
                            kl_loss = nn.KLDivLoss(reduction="mean")
                            KLD=kl_loss(vec1FN,vec2).item() #pred, true
                            KL1=kl_loss(vec1FN[:208],vec2[:208]).item()
                            KL2=kl_loss(vec1FN[208:1414],vec2[208:1414]).item()
                            KL3=kl_loss(vec1FN[1414:1514],vec2[1414:1514]).item()
                            KL4=kl_loss(vec1FN[1514:2254],vec2[1514:2254]).item()
                            KL5=kl_loss(vec1FN[2254:],vec2[2254:]).item()
                            #print("KLD :",KLD,KL1,KL2,KL3,KL4,KL5)
                            DF.at[track,"KL divergence FN"]=KLD
                            DF.at[track,"KL 1 FN"]=KL1
                            DF.at[track,"KL 2 FN"]=KL2
                            DF.at[track,"KL 3 FN"]=KL3
                            DF.at[track,"KL 4 FN"]=KL4
                            DF.at[track,"KL 5 FN"]=KL5
    
                            WSD=wasserstein_distance(vec1FN,vec2)
                            WS1=wasserstein_distance(vec1FN[:208],vec2[:208])
                            WS2=wasserstein_distance(vec1FN[208:1414],vec2[208:1414])
                            WS3=wasserstein_distance(vec1FN[1414:1514],vec2[1414:1514])
                            WS4=wasserstein_distance(vec1FN[1514:2254],vec2[1514:2254])
                            WS5=wasserstein_distance(vec1FN[2254:],vec2[2254:])
                            #print("WSD :",WSD,WS1,WS2,WS3,WS4,WS5)
    
                            DF.at[track,"WSD FN"]=WSD
                            DF.at[track,"WS 1 FN"]=WS1
                            DF.at[track,"WS 2 FN"]=WS2
                            DF.at[track,"WS 3 FN"]=WS3
                            DF.at[track,"WS 4 FN"]=WS4
                            DF.at[track,"WS 5 FN"]=WS5
                            for name, param in mod.named_parameters():
                                if name == 'vec2neck.weight':
                                    W = param
                                    break
                            CL=loss_Contractive(mod.vec2neck.weight,vec1FN,vec2, output[1], 0.00001)        
                            #print("Contractive :",CL.cpu().item())
                            DF.at[track,"contractive distance FN"]=CL.cpu().item()
                            Norm1=np.sum(np.abs(vec1FN.numpy()))
                            N11=np.sum(np.abs(vec1FN[:208].numpy()))
                            N12=np.sum(np.abs(vec1FN[208:1414].numpy()))
                            N13=np.sum(np.abs(vec1FN[1414:1514].numpy()))
                            N14=np.sum(np.abs(vec1FN[1514:2254].numpy()))
                            N15=np.sum(np.abs(vec1FN[2254:].numpy()))
                            #print("Weight pred L1: ",Norm1, N11,N12,N13,N14,N15)
                            DF.at[track,"N1 FN"]=Norm1
                            DF.at[track,"N11 FN"]=N11
                            DF.at[track,"N12 FN"]=N12
                            DF.at[track,"N13 FN"]=N13
                            DF.at[track,"N14 FN"]=N14
                            DF.at[track,"N15 FN"]=N15
                            Norm2=np.sum(np.abs(vec2.numpy()))
                            N21=np.sum(np.abs(vec2[:208].numpy()))
                            N22=np.sum(np.abs(vec2[208:1414].numpy()))
                            N23=np.sum(np.abs(vec2[1414:1514].numpy()))
                            N24=np.sum(np.abs(vec2[1514:2254].numpy()))
                            N25=np.sum(np.abs(vec2[2254:].numpy()))
                            #print("Weight GT L1: ",Norm2, N21,N22,N23,N24,N25)
                            DF.at[track,"N2 FN"]=Norm2
                            DF.at[track,"N21 FN"]=N21
                            DF.at[track,"N22 FN"]=N22
                            DF.at[track,"N23 FN"]=N23
                            DF.at[track,"N24 FN"]=N24
                            DF.at[track,"N25 FN"]=N25
                            #print("Saturation in pred ",100*sum(1 for x in vec1FN if x > 0.95 or x<0.05)/len(vec1FN),"% \t saturaion in target",100*sum(1 for x in vec2 if x > 0.95 or x<0.05)/len(vec2),"%")
                            #print("LWLN ",LW(vec1FN,vec2).item())
                            DF.at[track,"saturated in pred FN(%)"]=100*sum(1 for x in vec1FN if x > 0.95 or x<0.05)/len(vec1FN)
                            DF.at[track,"saturated in GT FN(%)"]=100*sum(1 for x in vec2 if x > 0.95 or x<0.05)/len(vec2)
                            DF.at[track,"LWLN FN"]=LWs(vec1FN,vec2).cpu().item()
    
                            
                            track=track+1
                    
                    try :
                        print(f"{str(datetime.datetime.now())} begin topo")
                        # Define filter function – can be any scikit-learn transformer
                        filter_func = Projection(columns=[0,1])
                        # Define cover
                        cover = CubicalCover(n_intervals=12, overlap_frac=0.15)
                        # Choose clustering algorithm – default is DBSCAN
                        clusterer = DBSCAN()
        
                        # Configure parallelism of clustering step
                        n_jobs = 8
        
                        GT=DF[Cols[2:2466]]
                        dataG=GT.to_numpy()
                        #print(Cols[2],Cols[2466],Cols[4930])
                        PD=DF[Cols[2466:4930]]
                        dataP=PD.to_numpy()
        
                        FN=DF[Cols[4930:7394]]
                        dataF=FN.to_numpy()
                        # Initialise pipeline
                        pipe = make_mapper_pipeline(filter_func=filter_func,cover=cover,clusterer=clusterer,verbose=False,n_jobs=n_jobs)
        
        
                        plotly_params = {"node_trace": {"marker_colorscale": "Blues"}}
                        fig = plot_static_mapper_graph(pipe, dataG, color_data=dataG, plotly_params=plotly_params)
        
                        fig1 = go.Figure(fig)
                        #fig1.write_image(results_path+f"Tracking/Mapper GT Graph 10 test set batches {epoch}.png")
                        wandb.log({f"Mapper GTruth Graph 10 test set batches {epoch}": wandb.Plotly(fig1)})
        
        
        
                        fig = plot_static_mapper_graph(pipe, dataP, color_data=dataP, plotly_params=plotly_params)
        
                        fig1 = go.Figure(fig)
                        #fig1.write_image(results_path+f"Tracking/Mapper Pred Graph 10 test set batches {epoch}.png")
                        wandb.log({f"Mapper Pred Graph 10 test set batches {epoch}": wandb.Plotly(fig1)})
        
        
                        fig = plot_static_mapper_graph(pipe, dataF, color_data=dataF, plotly_params=plotly_params)
        
                        fig1 = go.Figure(fig)
                        #fig1.write_image(results_path+f"Tracking/Mapper FN Graph 10 test set batches {epoch}.png")
                        wandb.log({f"Mapper Finetuned Graph 10 test set batches {epoch}": wandb.Plotly(fig1)})
                        print(f"{str(datetime.datetime.now())} end topo")
                    except Exception as e:
                        print(f"{str(datetime.datetime.now())} {e}")
                        continue 
                    wandb.log({"Epoch 0 Test CNN accuracy on 10 batches ": DF['epoch 0'].mean()})
                    wandb.log({"Epoch 1 Test CNN accuracy on 10 batches ": DF['epoch 1'].mean()})
                    # wandb.log({"Epoch 2 Test CNN accuracy on 10 batches ": DF['epoch 2'].mean()})
                    # wandb.log({"Epoch 3 Test CNN accuracy on 10 batches ": DF['epoch 3'].mean()})
                    # wandb.log({"Epoch 4 Test CNN accuracy on 10 batches ": DF['epoch 4'].mean()})
    
    
    
                if (epoch+1)%50==0:
                    DF.to_csv(results_path+f"Tracking/AE epoch {combination} {epoch}.csv")
                    torch.save({'epoch':epoch,'model_state_dict': mod.state_dict(),
                                'optimizerENC1_state_dict':  optimizerEnc1.state_dict() ,
                                'optimizerENC2_state_dict':optimizerEnc2.state_dict(),
                                'optimizerDense_state_dict':optimizerDense.state_dict(),
                                'optimizerDec_state_dict': optimizerDec.state_dict(),
                                'Batch Loss':used_Loss.detach().cpu().item(),
                                'validation entire testset MSE':loss_value.detach().cpu().item()},
                                results_path+'AE epoch{} {} {}.pth'.format(combination,track,epoch))
                    torch.cuda.empty_cache()
                if (epoch+1)==num_epochs:
                    del(mod)
                    del(optimizerEnc1)
                    del(optimizerEnc2) 
                    del(optimizerDense)
                    del(optimizerDec)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
                
            wandb.finish()
            track=+1
        #     return loss_tr
        # study= optuna.create_study(direction="minimize",storage=storage)
        # study.optimize(objective,n_trials=2,callbacks=[lambda study, trial: gc.collect()]+[logging_callback]) 
