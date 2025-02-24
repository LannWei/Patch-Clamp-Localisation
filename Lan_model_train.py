import torch, os, sys, argparse
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from Lan_model import UNet,ResNet18, ResNet50, ViT
from skimage.feature import peak_local_max
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

# child_dir = os.path.join(os.getcwd(), "Alex_pipette_tracking", "micropipette_tracker","preprocessing")
# os.chdir(child_dir)
from Alex_pipette_tracking.micropipette_tracker.preprocessing.data_loading import load_all_data, load_data_for_model
from Alex_pipette_tracking.micropipette_tracker.preprocessing.torch_loader import data_loader
from Alex_pipette_tracking.micropipette_tracker.preprocessing.data_loading import plot_images_with_labels
from Alex_pipette_tracking.micropipette_tracker.preprocessing.data_albumentations import apply_augmentmentation

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='exp')
parser.add_argument('--model', type=str, default='unet') #
parser.add_argument('--bs', type=int, default=64) # batch size
parser.add_argument('--heatmap_sigma', type=int, default=20) # heatmap radius
parser.add_argument('--lr', type=float, default=1e-4)  #0.0003 eu_dist
parser.add_argument('--eu_dist', type=int, default=10)  # 
parser.add_argument('--patience', type=int, default=20)  #50 
parser.add_argument('--n_epochs', type=int, default=150)  #100
parser.add_argument('--h2c_r', type=float, default=0.1)  # heatmap2coordinate threshold
# parser.add_argument('--fold', type=str, default='0')
parse_config = parser.parse_args()
print(parse_config)

exp_name = parse_config.exp_name + '_{}_r{}_d{}_lr{}_h2cr{}'.format(parse_config.model, parse_config.heatmap_sigma, parse_config.eu_dist, parse_config.lr, parse_config.h2c_r)
os.makedirs('logs/{}'.format(exp_name), exist_ok=True)
os.makedirs('logs/{}/model'.format(exp_name), exist_ok=True)
os.makedirs('logs/{}/image'.format(exp_name), exist_ok=True)
print("exp_name: ", exp_name)

writer = SummaryWriter('logs/{}/log'.format(exp_name))
best_path = 'logs/{}/model/best.pkl'.format(exp_name)
latest_path = 'logs/{}/model/latest.pkl'.format(exp_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

excel_files = [
      'Tip coordinates 19_07_23.xlsx', 'Tip coordinates 20_07_23.xlsx',
      'Tip coordinates 14_08_23.xlsx', 'Tip coordinates 03_08_23.xlsx',
      'Tip coordinates 30_08_23.xlsx', '31_08_23.xlsx', '01_09_23.xlsx']
directories = [
    r"Pics", r"20_07_23", r"14_08_23", r"03_08_23",
    r"30_08_23", r"31_08_23", r"01_09_23"]


class KeypointDataset(Dataset):
    def __init__(self, labels, images, img_size=256, sigma=2):
        self.labels = labels
        self.images = images
        self.img_size = img_size
        self.sigma = sigma
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size))])

    def gaussian_heatmap(self, height, width, center, sigma):
        heatmap = np.zeros((height, width), dtype=np.float32)
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        x0, y0 = center
        heatmap = np.exp(-((x - x0)**2 + (y - y0)**2) / (sigma**2))
        return heatmap

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(image)
        keypoints = self.labels[idx]
        heatmaps = np.zeros((self.img_size, self.img_size))
        for kp in keypoints:
            heatmap = self.gaussian_heatmap(self.img_size, self.img_size, (kp[0], kp[1]), self.sigma)
            heatmaps += heatmap
        
        heatmaps = torch.tensor(heatmaps).unsqueeze(0)
        return image, heatmaps


def check_point_is_correct(pred: list, actual: list, eu_dist) -> bool:
    """Check if the predicted point is within a certain euclidean distance of the actual point."""
    return ((pred[0] - actual[0])**2 + (pred[1] - actual[1])**2)**0.5 <= eu_dist

def hungarian_algorithm(preds: list, actuals: list) -> list:
    """Use the Hungarian algorithm to match predicted points to actual points."""
    cost_matrix = np.linalg.norm(preds[:, None] - actuals[None, :], axis=-1)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches = list(zip(row_indices, col_indices))
    return matches

def heatmap_to_coordinates(heatmap, threshold=0.1) -> list:
    """Convert a heatmap to a list of coordinates."""
    coordinates = peak_local_max(heatmap, min_distance=20, threshold_abs=threshold, exclude_border=False)
    return coordinates

def plot_results(image, pred_heatmap, true_keypoints, pred_keypoints, matches, incorrect_matches):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image[0], cmap='gray')
    ax.imshow(pred_heatmap, cmap='jet', alpha=0.5)  # Adjust alpha for visibility
    ax.scatter(true_keypoints[:, 0], true_keypoints[:, 1], c='green', marker='o', label='True Keypoints')
    correct_matches = [match for match in matches if match not in incorrect_matches]
    correct_pred_points = pred_keypoints[[match[0] for match in correct_matches]]
    ax.scatter(correct_pred_points[:, 0], correct_pred_points[:, 1], c='blue', marker='x', label='Correct Predictions')
    incorrect_pred_points = pred_keypoints[[match[0] for match in incorrect_matches]]
    ax.scatter(incorrect_pred_points[:, 0], incorrect_pred_points[:, 1], c='red', marker='x', label='Incorrect Predictions')
    ax.legend()
    # Convert plot to an image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)

    # Convert buffer to PIL Image (NOT a NumPy array)
    img = Image.open(buf)
    
    return img  # Return the PIL Image

def correct_percentage(model, images, labels, mode):
    model.eval()
    total_point = 0
    incorrect_point = 0
    with torch.no_grad():
        for idx in range(0, len(images), parse_config.bs):
            if idx+parse_config.bs > len(images):
                break
            image = images[idx:idx+parse_config.bs] # batch data
            image = np.array(image)
            image_tensor = torch.tensor(image).transpose(1, 3).transpose(2, 3).to(device)
            preds = model(image_tensor)  # [16, 1, 256, 256]
            preds = preds.squeeze().cpu().detach().numpy() # (16, 256, 256)
            
            for i, pred in enumerate(preds):
                coordinates = heatmap_to_coordinates(pred, parse_config.h2c_r) #(17, 2)
                coordinates = coordinates[:, [1, 0]] #(17, 2) change x and y value
                actuals = np.array(labels[idx+i])
                total_point += actuals.shape[0]
                if len(coordinates) == 0 or len(actuals) == 0:
                    # print("***0***: ", len(coordinates), len(actuals))
                    incorrect_point += len(actuals)
                    continue
                try:
                    matches = hungarian_algorithm(coordinates, actuals)
                    incorrect_matches = [match for match in matches if not check_point_is_correct(coordinates[match[0]], actuals[match[1]], parse_config.eu_dist)]
                    if len(matches) != len(actuals):
                        incorrect_point += abs(len(matches) - len(actuals))
                    for match in matches:
                        if not check_point_is_correct(coordinates[match[0]], actuals[match[1]], parse_config.eu_dist):
                            incorrect_point += 1
                            
                    if mode == 'test':        
                        plotted_img = plot_results(image_tensor.cpu().detach().numpy()[i], pred, actuals, coordinates, matches, incorrect_matches)
                        image_path = 'logs/{}/image/test_image{}.png'.format(exp_name,i+idx)
                        plotted_img.save(image_path)
                        print(f"Saved: {image_path}")   
                        
                except Exception as e:
                    print("Error occured", e)
        Correct_percentage = (total_point - incorrect_point) / total_point * 100

    return Correct_percentage

def train_model(model, train_dataloader, val_dataloader,  loss_fn, device, num_epochs):   
    model = model.to(device)
    best_per = 0 
    min_loss = 10
    best_ep = 0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    
    for epoch in range(1, num_epochs+1):
        valid_epoch_loss = 0
        this_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('Learning Rate', this_lr, epoch)
        
        model.train()

        for batch_idx, (images, heatmaps) in enumerate(train_dataloader):
        # for images, heatmaps in train_dataloader:
            images, heatmaps = images.to(device, dtype=torch.float32), heatmaps.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            pred_heatmaps = model(images)
            # print("*** input image shape", heatmaps.shape)
            loss = loss_fn(pred_heatmaps, heatmaps)
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 10 == 0:
                writer.add_scalar('loss/loss', loss, batch_idx + epoch * len(train_dataloader))
                writer.add_image('Train/heatmaps', heatmaps[0], batch_idx + epoch * len(train_dataloader))
                writer.add_image('Train/pred_heatmaps', pred_heatmaps[0], batch_idx + epoch * len(train_dataloader))

                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, parse_config.n_epochs, batch_idx * len(images),
                        len(train_dataloader.dataset), 100. * batch_idx / len(train_dataloader), loss.item()))
        scheduler.step()

        
        model.eval()
        with torch.no_grad():
            for images, heatmaps in val_dataloader:
                images, heatmaps = images.to(device, dtype=torch.float32), heatmaps.to(device, dtype=torch.float32)
                pred_heatmaps = model(images)
                loss = loss_fn(pred_heatmaps, heatmaps)
                valid_epoch_loss += loss.item()
        
        val_epoch_loss = valid_epoch_loss / len(valid_dataset)
        Correct_percentage = correct_percentage(model, val_images, val_labels, 'val')
        writer.add_scalar('Val/loss', val_epoch_loss, epoch)
        writer.add_scalar('Val/perncentage', Correct_percentage, epoch)
        print('Val Epoch: {}/{} Loss: {:.6f}'.format(epoch, parse_config.n_epochs, val_epoch_loss), 'Correct_per: {:.4f}'.format(Correct_percentage))

        if val_epoch_loss < min_loss:
            min_epoch = epoch
            min_loss = loss
        else:
            if epoch - min_epoch >= parse_config.patience:
                print('Early stopping!')
                break            

        if Correct_percentage > best_per:
            best_per = Correct_percentage
            best_ep = epoch
            torch.save(model.state_dict(), best_path)
            print('Update the best model: epoch {} with precentage {:.4f}'.format(epoch, Correct_percentage))
        else:
            if epoch - best_ep >= parse_config.patience:
                print('Early stopping!')
                break
        torch.save(model.state_dict(), latest_path)
    
    # print('The highest accuracy achieved by U-Net is {:.4f}'.format(best_per))
    print('The highest accuracy achieved by {} is {:.4f}'.format(parse_config.model, best_per))

def test_model(model, test_images, test_labels, device):
    model.load_state_dict(torch.load(best_path))
    model = model.to(device)

    Correct_percentage = correct_percentage(model, test_images, test_labels,'test')
    print('Test keypoint prediction correct percentage: {:.4f}'.format(Correct_percentage))



###################################################################
all_images, all_labels = load_all_data(excel_files, directories)
print("Before augmentation: we have ", len(all_images), " images and ", len(all_labels), " labels.")

# Get test image without augmentation
test_ratio = 0.2
temp_images, test_images, temp_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=test_ratio, random_state=42)
test_labels = [[(label[i], label[i+1]) for i in range(0, len(label), 2) if label[i] != 0 and label[i+1] != 0] for label in test_labels]
print("Train and val images: ", len(temp_images), "Test images:", len(test_images))

# Get augmented train and val image
temp_labels = [[(label[i], label[i+1]) for i in range(0, len(label), 2) if label[i] != 0 and label[i+1] != 0] for label in temp_labels]
all_augmented_images = []
all_augmented_labels = []
for i in range(20):
    augmented_images, augmented_labels = apply_augmentmentation(temp_images, temp_labels)
    all_augmented_images.extend(augmented_images)
    all_augmented_labels.extend(augmented_labels)
# print("Train and val after augmentation: ", len(all_augmented_images), " images and ", len(all_augmented_labels), " labels.")
val_ratio = 0.15
train_images, val_images, train_labels, val_labels = train_test_split(all_augmented_images, all_augmented_labels, test_size=val_ratio, random_state=42)

# Train and Val dataset
training_dataset = KeypointDataset(train_labels, train_images, sigma=parse_config.heatmap_sigma)
valid_dataset    = KeypointDataset(val_labels,   val_images,   sigma=parse_config.heatmap_sigma)
train_dataloader = DataLoader(training_dataset, batch_size=parse_config.bs, shuffle=True)
val_dataloader   = DataLoader(valid_dataset,    batch_size=parse_config.bs, shuffle=True)
print("After augmentation total: ", len(all_augmented_images), "Train images:", len(train_images), "Val images:", len(val_images))

if parse_config.model == 'unet':
    model = UNet()
elif parse_config.model == 'resnet18':
    model = ResNet18()
elif parse_config.model == 'resnet50':
    model = ResNet50()
elif parse_config.model == 'vit':
    model = ViT()
else:
    "Wrong Input Model!"
    
loss_fn = nn.SmoothL1Loss()

train_model(model, train_dataloader, val_dataloader, loss_fn, device=device, num_epochs=parse_config.n_epochs)

test_model(model, test_images, test_labels, device=device)

writer.close()
