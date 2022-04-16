#1
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import time
import torchvision.utils as vutils
import torch.nn.functional as F

from torchvision import datasets, transforms
from draw_model import DRAWModel
from dataloader import get_data
import sys
import os


import argparse
parser = argparse.ArgumentParser(description="Template")
# Dataset options
parser.add_argument('-ep', '--epochs', default="100", help="training epochs")
parser.add_argument('-m', '--multiplex', default="3", help="multiplex_num")
parser.add_argument('-gpu', '--gpu', default="0", help="gpu number")
parser.add_argument('-z', '--z-dim', default="100", help="z")
parser.add_argument('-g', '--glimpses', default="64", help="glimpse")
parser.add_argument('-ds', '--dataset', default="mnist", help="mnist,svhn")
parser.add_argument('-cf', '--checkfolder', default="./checkfolder/", help="check folder")
# Parse arguments
opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    # print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\n')
    s = '{} |{}| {}% {}'.format(prefix,bar,percent,suffix)
    sys.stdout.write(' ' * (s.__len__()+3) + '\r')
    sys.stdout.flush()
    sys.stdout.write(s + '\r')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()

# Function to generate new images and save the time-steps as an animation.
def generate_image(epoch):
    x = model.generate(64)
    fig = plt.figure(figsize=(16, 16))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in x]
    anim = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000, blit=True)
    anim.save('draw_epoch_{}.gif'.format(epoch), dpi=100, writer='imagemagick')
    plt.close('all')



def get_inputx(x_input_list, multi):
    X_input = x_input_list[0]
    for X_input_i in x_input_list[1:]:
        X_input = torch.cat((X_input, X_input_i),1)

    for x in range(multi - len(x_input_list)):
        X_input = torch.cat((X_input, torch.zeros_like(x_input_list[0])),1)
    return X_input

def draw_encode_decode(x, model, multi):
    x_input_list = [x]

    for i in range(multi - 1):  
        x_input = get_inputx(x_input_list, multi)
        recon_x = model(x_input)
        x_input_list.append((x - recon_x)/2)


    x_input = get_inputx(x_input_list, multi)
    recon_x = model(x_input)
    return recon_x


def eval_draw(test_loader, model, multi, device, params):
    log_loss_a = 0


    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):

            bs = x.size(0)
            x = x.to(device).view(bs,-1)
            #recon_x = model(x)
            recon_x = draw_encode_decode(x, model, multi)

            log_loss_a += F.mse_loss(recon_x.view(bs,-1,params['A'],params['B']), x.view(bs,-1,params['A'],params['B']))
            #log_loss_a += -psnr(recon_x, x)

    return (log_loss_a/len(test_loader)).item()

# Dictionary storing network parameters.
params = {
    'T' : int(opt.glimpses),# Number of glimpses.
    'A' : 0,# Image width
    'B': 0,# Image height
    'z_size' :int(opt.z_dim),# Dimension of latent space.
    'read_N' : 0,# N x N dimension of reading glimpse.
    'write_N' : 0,# N x N dimension of writing glimpse.
    'dec_size': 0,# Hidden dimension for decoder.
    'enc_size' :0,# Hidden dimension for encoder.
    'epoch_num': int(opt.epochs),# Number of epochs to train for.
    'learning_rate': 1e-3,# Learning rate.
    'beta1': 0.5,
    'clip': 5.0,
    'channel' : None,

    'multi': int(opt.multiplex),
    }# Number of channels for image.(3 for RGB, etc.)

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")



params['device'] = device

'''
train_loader = get_data(params)
params['channel'] = 3
'''

if opt.dataset == 'mnist':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/work/tp/NearLoss/data/MNIST/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()])),
        batch_size=128, shuffle=True,num_workers=16)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/home/work/tp/NearLoss/data/MNIST/', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()])),
        batch_size=128, shuffle=True,num_workers=16)
    params['channel'] = 1
    params['A'] = 28
    params['B'] = 28
    params['read_N'] = 2
    params['write_N'] = 5
    params['dec_size'] = 256
    params['enc_size'] = 256
#     params['T'] = 32

elif opt.dataset == 'Parabola':
    import tp_util as tp
    img_size = 64
    dataset_saved_dir = '/home/work/tp/NearLoss/data/Blade/processed/512x512/Parabola/'
    transform=transforms.Compose([transforms.Resize(size=img_size),
                                transforms.ToTensor()])
    trainset = tp.Blade_dataset(file_path=dataset_saved_dir + 'train.pt', transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers=2)
    testset = tp.Blade_dataset(file_path=dataset_saved_dir + 'test.pt', transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                              shuffle=True, num_workers=2)
    params['channel'] = 1
    params['A'] = img_size
    params['B'] = img_size
    params['read_N'] = 5
    params['write_N'] = 5
    params['dec_size'] = 256
    params['enc_size'] = 256
#     params['T'] = 32

elif opt.dataset in ['SVHN', 'svhn']:
    batch_size = 128
    transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('/home/work/tp/NearLoss/data/SVHN/', split='train', 
                    transform=transform, download=True),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('/home/work/tp/NearLoss/data/SVHN/', split='test', 
                    transform=transform, download=True),
        batch_size=batch_size, shuffle=True)
    params['channel'] = 1
    params['A'] = 32
    params['B'] = 32
    params['read_N'] = 5
    params['write_N'] = 5
    params['dec_size'] = 256
    params['enc_size'] = 256
#     params['T'] = 32

'''
# Plot the training images.
sample_batch = next(iter(train_loader))
plt.figure(figsize=(16, 16))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 64], nrow=8, padding=1, normalize=True, pad_value=1).cpu(), (1, 2, 0)))
plt.savefig("Training_Data")
'''


# Initialize the model.
model = DRAWModel(params).to(device)
# Adam Optimizer
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], betas=(params['beta1'], 0.999))

# List to hold the losses for each iteration.
# Used for plotting loss curve.
losse_train_list = []
losse_test_list = []

print("-"*25)
print(opt)
print(params)
print("Starting Training Loop...\n")
print('Epochs: %d\nLength of Data Loader: %d' % (params['epoch_num'], len(train_loader)))
print("-"*25)

eval_draw(test_loader, model, multi=params['multi'], device=device, params=params)
plt.figure(figsize=(10,5))

for epoch in range(params['epoch_num']):
    loss_a = 0
    
    for i, (data, _ ) in enumerate(train_loader, 0):
        printProgressBar(i+1, len(train_loader))
        # Get batch size.
        bs = data.size(0)
        # Flatten the image.
        data = data.to(device).view(bs, -1)
        optimizer.zero_grad()
        # Calculate the loss.
        draw_encode_decode(data, model, multi=params['multi'])
        loss = model.loss(data)
        loss_a += loss
        # Calculate the gradients.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
        # Update parameters.
        optimizer.step()

    loss_train = loss_a.item() / len(train_loader)
    losse_train_list.append(loss_train)
    loss_val = eval_draw(test_loader, model, multi=params['multi'], device=device, params=params)
    losse_test_list.append(loss_val)
    print("Epoch: {:03d}/{:03d}, Train_Loss: {:.2f}, Test_MSE: {:.5f}".format(epoch+1, params['epoch_num'], loss_train, loss_val))
    
    # Plot the training losses.
    
    plt.title("Training Loss")
    plt.plot(losse_train_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f"{opt.checkfolder}/{opt.dataset}_TrainLoss_z-{params['z_size']}_multi-{params['multi']}_T-{params['T']}")
    
    plt.cla()
    plt.title("Test loss")
    plt.plot(losse_test_list)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.savefig(f"{opt.checkfolder}/{opt.dataset}_MSE_z-{params['z_size']}_multi-{params['multi']}_T-{params['T']}")
    
    if os.path.exists(f"{opt.checkfolder}/stop"):
        break;

losse_test_list = np.array(losse_test_list)
print(f'min epoch: {losse_test_list.argmin()+1}')
print(f'min loss: {losse_test_list.min()}')

    

# Save the final trained network paramaters.
'''
torch.save({
    'model' : model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'params' : params
    }, 'checkpoint/model_final'.format(epoch))
'''
# Generate test output.
'''
with torch.no_grad():
    generate_image(params['epoch_num'])
'''



file = f'{opt.checkfolder}/log.txt'
with open(file, 'a+') as f:
    f.write(str(params))
    f.write('\n{}, z-{}_m{}, min MSE: {:.8f}, min epoch: {}/{}; Min train loss: {:.2f}\n\n'.format(
        opt.dataset, params['z_size'], params['multi'], float((losse_test_list).min()), 
        losse_test_list.argmin()+1, params['epoch_num'], float(np.array(losse_train_list).min())))