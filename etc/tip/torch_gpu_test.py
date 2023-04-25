import torch

if torch.cuda.is_available():
    
    num = torch.cuda.device_count()

    for i in range(num):
        print("gpu" + str(i) + "\t" + str(torch.cuda.get_device_name(i)))

else:
    print("there is no gpu")

