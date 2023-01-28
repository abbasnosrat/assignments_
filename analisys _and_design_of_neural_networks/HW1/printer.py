import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")
dir_list = ["./YOLOX/YOLOX_outputs/yolox_voc_s/train_log.txt","./YOLOX/YOLOX_outputs/yolox_voc_lcn/train_log.txt"]
for dir in dir_list:
    file = open(dir,"r")
    loss_list = []
    for line in file:
        if "total_loss" in line:
        
            l = line.split(",")
            loss_str = l[5]
            loss_str = loss_str.split(" ")
            loss = float(loss_str[2])
            loss_list.append(loss)
    loss_mean = []
    loss_se = []
    step = 1000
    for i in range(len(loss_list)-step):
        
       
        loss_mean.append(np.mean(loss_list[i:i+1253]))
        loss_se.append(np.std(loss_list[i:i+1253])/np.sqrt(1253))
    loss_mean = np.array(loss_mean)
    loss_se = np.array(loss_se)
    plt.plot(loss_mean)
    plt.fill_between(np.arange(len(loss_mean)),loss_mean+loss_se,loss_mean-loss_se,alpha = 0.5)
    #plt.errorbar(list(range(len(loss_se))),loss_mean,loss_se)
                   
       
    #plt.plot(loss_list)
plt.legend(["base",None,"with lcn",None])
plt.title(f"loss decrease of YOLOX and YOLOX with LCN for every {step} iteration")
plt.show()

