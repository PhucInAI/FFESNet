import os
import glob
import shutil
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import imageio
from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")
import logging
import matplotlib.pyplot as plt

from FFESNet.utils.dataset import PolypDataset, TestDataset
from FFESNet.utils.losses.structure_loss import structure_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments of Learning Ability training pipeline')

    parser.add_argument('--config', '-c', type=str, required=True       , help='Path of the config file')
    parser.add_argument('--device', '-d', type=int, required=True       , help='Device to use')
    parser.add_argument('--output', '-o', type=str, default='./runs'    , help='Path of ouput folder')

    args = parser.parse_args()

    return args


def load_loss(loss_name):
    """Load loss function"""

    if loss_name == 'structure':
        from FFESNet.utils.losses.structure_loss import structure_loss as loss_function
    elif loss_name == 'dice_bce':
        from FFESNet.utils.losses.dice_bce_loss import dice_bce_loss as loss_function
    elif loss_name == 'tversky':
        from FFESNet.utils.losses.tversky_loss import tversky_loss as loss_function
    elif loss_name == 'tversky_bce':
        from FFESNet.utils.losses.tversky_bce_loss import tversky_bce_loss as loss_function
    elif loss_name == 'sufl':
        from FFESNet.utils.losses.loss_classes import SymmetricUnifiedFocalLoss
        loss_function = SymmetricUnifiedFocalLoss()
    else:
        print('Do not know loss name')
        exit()

    return loss_function


def load_model(model_arch, model_type, drop_rate):
    """Load model"""

    if model_arch == 'ESFPNet':
        from FFESNet.models.ESFPNet import ESFPNet
        model = ESFPNet(model_type=model_type)
        return model

    if model_arch == 'FFESNet_BiFPN':
        from  FFESNet.models.FFESNet_BiFPN import FFESNet
    elif model_arch == 'FFESNet_BiFPN_AttentionAggregation':
        from FFESNet.models.FFESNet_BiFPN_AttentionAggregation import FFESNet
    elif model_arch == 'FFESNet_A2FPN':
        from FFESNet.models.FFESNet_A2FPN import FFESNet
    elif model_arch == 'FFESNet_LE_A2FPN':
        from FFESNet.models.FFESNet_LE_A2FPN import FFESNet

    model = FFESNet(model_type=model_type, dropout=drop_rate)
    
    return model


def evaluate(config, model):  
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ESFPNet.eval()
    
    global device

    val = 0
    count = 0
    smooth = 1e-4
    
    # --------------------------------------------------------------------
    # Set up data
    # --------------------------------------------------------------------
    init_trainsize = int(config['hyparameters']['init_trainsize'])
    val_path = config['dataset']['valid']
    val_images_path = os.path.join(val_path, 'images')
    val_masks_path = os.path.join(val_path, 'masks')
    val_loader = TestDataset(val_images_path,val_masks_path, init_trainsize)
    
    # --------------------------------------------------------------------
    # Validating pipleline
    # --------------------------------------------------------------------
    model.eval()

    for i in range(val_loader.size):
        image, gt, name = val_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.to(device)
        
        pred = model(image)
        pred = F.upsample(pred, size=gt.shape, mode='bilinear', align_corners = False)
        
        pred = pred.sigmoid()
        threshold = torch.tensor([0.5]).to(device)
        pred = (pred > threshold).float() * 1
        pred = pred.data.cpu().numpy().squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        
        target = np.array(gt)
        input_flat = np.reshape(pred,(-1))
        target_flat = np.reshape(target,(-1))
 
        intersection = (input_flat*target_flat)
        loss =  (2 * intersection.sum() + smooth) / (pred.sum() + target.sum() + smooth)

        a =  '{:.4f}'.format(loss)
        a = float(a)
        
        val = val + a
        count = count + 1
        
    model.train()
    
    return val/count


def save_result(numIters, config, model_path):
    global device
    global output_path

    save_path = os.path.join(output_path, str(numIters).zfill(2), 'predict')
    os.makedirs(save_path, exist_ok=True)

    # --------------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------------
    model = torch.load(model_path)
    model.eval()

    # --------------------------------------------------------------------
    # Set up data
    # --------------------------------------------------------------------
    init_trainsize = int(config['hyparameters']['init_trainsize'])
    test_path = config['dataset']['test']
    test_images_path = os.path.join(test_path, 'images')
    test_masks_path = os.path.join(test_path, 'masks')
    test_loader = TestDataset(test_images_path, test_masks_path, init_trainsize)
    

    # --------------------------------------------------------------------
    # Testing pipeline
    # --------------------------------------------------------------------
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device)

        pred = model(image)
        pred = F.upsample(pred, size=gt.shape, mode='bilinear', align_corners=False)
        pred = pred.sigmoid()
        threshold = torch.tensor([0.5]).to(device)
        pred = (pred > threshold).float() * 1
        pred = pred.data.cpu().numpy().squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        
        imageio.imwrite(os.path.join(save_path, name),img_as_ubyte(pred))


def plot_result(result_lst, save_path):
    epochs = range(1, len(result_lst) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, result_lst, label='Training Loss', marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(save_path)
    # plt.show()


def train_loop(config, numIters):
    global device
    global output_path

    model_type = config['model']['model_type']
    model_arch = config['model']['model_arch']

    drop_rate = float(config['hyparameters']['drop_rate'])
    n_epochs = int(config['hyparameters']['n_epochs'])
    init_lr = float(config['hyparameters']['learning_rate'])
    init_trainsize = int(config['hyparameters']['init_trainsize'])
    batch_size = int(config['hyparameters']['batch_size'])

    # --------------------------------------------------------------------
    # Setup log
    # --------------------------------------------------------------------
    # Set up the logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create a file handler that writes log messages to a file
    file_handler = logging.FileHandler(os.path.join(output_path, 'logfile_{}.log'.format(str(numIters).zfill(2))))
    file_handler.setLevel(logging.INFO) 
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Create a stream handler that writes log messages to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    # Get the root logger and add the handlers
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # --------------------------------------------------------------------
    # Clear GPU cache and load model
    # --------------------------------------------------------------------
    torch.cuda.empty_cache()
    model = load_model(model_arch, model_type, drop_rate)
    model.to(device)
    lr = init_lr
    
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=25, gamma=0.25)
    
    # --------------------------------------------------------------------
    # Keep track of losses over time
    # --------------------------------------------------------------------
    losses = []
    coeff_max = 0
    
    # --------------------------------------------------------------------
    # Set up data
    # --------------------------------------------------------------------
    train_path = config['dataset']['train']
    train_images_path = os.path.join(train_path, 'images')
    train_masks_path = os.path.join(train_path, 'masks')
    trainDataset = PolypDataset(train_images_path, train_masks_path, trainsize = init_trainsize, augmentations = True)
    train_loader = DataLoader(dataset = trainDataset, batch_size = batch_size, shuffle = True)
    
    iter_X = iter(train_loader)
    steps_per_epoch = len(iter_X)
    num_epoch = 0
    total_steps = (n_epochs+1)*steps_per_epoch

    # --------------------------------------------------------------------
    # Training pipleline
    # --------------------------------------------------------------------
    for step in range(1, total_steps):

        # ----------------------------------------------------------------
        # Reset iterators for each epoch
        # ----------------------------------------------------------------
        if step % steps_per_epoch == 0:
            iter_X = iter(train_loader)
            num_epoch = num_epoch + 1
        
        # ----------------------------------------------------------------
        # Compute loss in 1 step
        # ----------------------------------------------------------------
        images, masks = next(iter_X)
        images = images.to(device)
        masks = masks.to(device)
       
        model.zero_grad()
        out = model(images)
        out = F.interpolate(out, size=masks.shape[2:], mode='bilinear', align_corners=False)
        
        if 'loss' in config['hyparameters']:
            loss_name = config['hyparameters']['loss']
        else:
            loss_name = 'structure'
        loss_function = load_loss(loss_name)
        loss = loss_function(out, masks)
        loss.backward()
        model_optimizer.step() 
        
        # ----------------------------------------------------------------
        # Validate each epoch
        # ----------------------------------------------------------------
        # Print the log info
        if step % steps_per_epoch == 0:
            # ------------------------------------------------------------
            # Log loss of train
            losses.append(loss.item())
            logger.info('Epoch [{:5d}/{:5d}] | preliminary loss: {:6.6f} '.format(num_epoch, n_epochs, loss.item()))

            # ------------------------------------------------------------
            # Update lr
            scheduler.step()

            # ------------------------------------------------------------
            # Log loss of valid
            validation_coeff = evaluate(config, model)
            logger.info('Epoch [{:5d}/{:5d}] | validation coeffient: {:6.6f} '.format(num_epoch, n_epochs, validation_coeff))
            
            # ------------------------------------------------------------
            # Save model if get better validation
            if coeff_max < validation_coeff:
                coeff_max = validation_coeff
                
                save_model_folder = os.path.join(output_path, str(numIters).zfill(2))
                os.makedirs(save_model_folder, exist_ok=True)
                save_model_path = os.path.join(save_model_folder, 'model_{}_{}.pt'.format(numIters, num_epoch))
                torch.save(model, save_model_path)
                logger.info('Save Learning Ability Optimized Model at Epoch [{:5d}/{:5d}]'.format(num_epoch, n_epochs))


    save_result(numIters, config, save_model_path)
    loss_path = os.path.join(output_path, 'loss_{}.png'.format(str(numIters)).zfill(2))
    plot_result(losses, loss_path)     
    return losses, coeff_max


def train_repeats(config, output_dir):
    """Train pipeline 1 repeat"""
    global device
    global output_path

    model_name = config['model']['model_name']
    repeats = int(config['hyparameters']['repeats'])
    # --------------------------------------------------------------------
    # Prepare folder
    # --------------------------------------------------------------------
    output_path = os.path.join(output_dir, model_name)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    # --------------------------------------------------------------------
    # Repeats pipline
    # --------------------------------------------------------------------
    for i in range(repeats):
        losses, coeff_max = train_loop(config, i+1)


def main():
    args = parse_args()

    global device
    global config

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    with open(args.config, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    train_repeats(config, args.output)


if __name__=="__main__":
    main()