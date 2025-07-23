import cv2
import itertools
import numpy as np
import torch.nn as nn
from PIL import Image
from io import BytesIO
import albumentations as A
import matplotlib.pyplot as plt
import torch, os, sys, argparse
import torchvision.transforms as T
from collections import OrderedDict
from skimage.feature import peak_local_max
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from Lan_model_co2fi_inference import ViTHeatmapKeypointModel,ViTHeatmapModel,UNet
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from Alex_pipette_tracking.micropipette_tracker.preprocessing.data_loading import load_all_data, load_data_for_model
from Alex_pipette_tracking.micropipette_tracker.preprocessing.torch_loader import data_loader
from Alex_pipette_tracking.micropipette_tracker.preprocessing.data_loading import plot_images_with_labels

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='reg')
parser.add_argument('--model', type=str, default='vit') #
parser.add_argument('--bs', type=int, default=32) # batch size
parser.add_argument('--heatmap_sigma', type=float, default=10) # heatmap radius
parser.add_argument('--lr', type=float, default=2.1e-5)  #0.0003 eu_dist
parser.add_argument('--eu_dist', type=int, default=10)  # parse_config.eu_dist
parser.add_argument('--patience', type=int, default=50)  #50 
parser.add_argument('--n_epochs', type=int, default=150)  #100
parser.add_argument('--h2c_r', type=float, default=0.1)  # heatmap2coordinate threshold
parser.add_argument('--gan_enhance', type=int, default=0) # batch size
parser.add_argument('--max_kp', type=int, default=4) # parse_config.lr
parse_config = parser.parse_args()
print(parse_config) # 

exp_name = parse_config.exp_name + '_{}_r{}_d{}_gan{}_lr{}'.format(parse_config.model, parse_config.heatmap_sigma, parse_config.eu_dist, parse_config.gan_enhance, parse_config.lr)
os.makedirs('logs_new/{}'.format(exp_name), exist_ok=True)
os.makedirs('logs_new/{}/model'.format(exp_name), exist_ok=True)
os.makedirs('logs_new/{}/image'.format(exp_name), exist_ok=True)
print("exp_name: ", exp_name)

writer = SummaryWriter('logs_new/{}/log'.format(exp_name))
best_path = 'logs_new/{}/model/best.pkl'.format(exp_name)
latest_path = 'logs_new/{}/model/latest.pkl'.format(exp_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

excel_files = [
      'Tip coordinates 19_07_23.xlsx', 'Tip coordinates 20_07_23.xlsx',
      'Tip coordinates 14_08_23.xlsx', 'Tip coordinates 03_08_23.xlsx',
      'Tip coordinates 30_08_23.xlsx', '31_08_23.xlsx', '01_09_23.xlsx']
directories = [
    r"Pics", r"20_07_23", r"14_08_23", r"03_08_23",
    r"30_08_23", r"31_08_23", r"01_09_23"]

def load_gan_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")
    image = image.astype(np.float32) / 255.0
    image = image.reshape((256, 256, 1))
    return image

def augment_images_labels(images, labels, is_train=True):
    if is_train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.6, border_mode=cv2.BORDER_REPLICATE),
            A.Rotate(limit=30, p=0.25),
            A.RandomBrightnessContrast(p=0.25),
            A.Blur(p=0.20),
            A.CLAHE(p=0.20),
            A.AdvancedBlur(p=0.20),
            # A.AutoContrast(p=0.20),
            A.Defocus(p=0.15),
            A.Downscale(p=0.15),
            A.Sharpen(p=0.5),
            A.Resize(224, 224)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
    else:  # Validation - only resizing
        transform = A.Compose([
            A.Resize(224, 224)
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    augmented_images = []
    augmented_labels = []
    resized_heatmaps = []

    for img, kps in zip(images, labels):
        augmented = transform(image=img, keypoints=kps)
        
        transformed_kps = [(kp[0], kp[1]) for kp in augmented['keypoints']]
        augmented_images.append(augmented['image'])
        augmented_labels.append(transformed_kps)
    return augmented_images, augmented_labels

class KeypointDataset(Dataset):
    def __init__(self, images, labels, max_keypoints=parse_config.max_kp, img_size=224, sigma=2):
        self.labels = labels
        self.images = images
        self.img_size = img_size
        self.sigma = sigma
        self.max_keypoints = max_keypoints
        self.transform = T.Compose([
            T.ToTensor()])

    def __len__(self):
        return len(self.images)

    def gaussian_heatmap(self, height, width, center, sigma):
        heatmap = np.zeros((height, width), dtype=np.float32)
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        x0, y0 = center
        heatmap = np.exp(-((x - x0)**2 + (y - y0)**2) / (sigma**2))
        return heatmap

    def pad_keypoints(self, keypoints):
        num_kp = len(keypoints)
        if num_kp > self.max_keypoints:
            keypoints = keypoints[:self.max_keypoints]
        keypoints_tensor = torch.zeros((self.max_keypoints, 2), dtype=torch.float32) 
        if num_kp > 0:
            keypoints_tensor[:num_kp] = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints_tensor
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(image)
        keypoints = self.labels[idx]
        heatmaps = np.zeros((self.img_size, self.img_size))
        for kp in keypoints:
            heatmap = self.gaussian_heatmap(self.img_size, self.img_size, (kp[0], kp[1]), self.sigma)
            heatmaps += heatmap
        # keypoints.extend([(0, 0)] * (4 - len(keypoints)))
        heatmaps = torch.tensor(heatmaps).unsqueeze(0)
        pad_keypoints = self.pad_keypoints(keypoints)
        return image, heatmaps, pad_keypoints


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
    coordinates = peak_local_max(heatmap, min_distance=30, threshold_abs=threshold, exclude_border=False)
    return coordinates

def plot_results(image, pred_heatmap, true_keypoints, pred_keypoints, matches, incorrect_matches):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image[0], cmap='gray')
    ax.imshow(pred_heatmap, cmap='jet', alpha=0.5)  # Adjust alpha for visibility
    # ax.scatter(true_keypoints[:, 0], true_keypoints[:, 1], c='green', marker='o', s=100, label='True Coordinations')
    # correct_matches = [match for match in matches if match not in incorrect_matches]
    # correct_pred_points = pred_keypoints[[match[0] for match in correct_matches]]
    # ax.scatter(correct_pred_points[:, 0], correct_pred_points[:, 1], c='blue', marker='x', s=100, label='Correct Predictions')
    # incorrect_pred_points = pred_keypoints[[match[0] for match in incorrect_matches]]
    # ax.scatter(incorrect_pred_points[:, 0], incorrect_pred_points[:, 1], c='red', marker='x', s=100, label='Incorrect Predictions')
    # ax.legend(fontsize=18)
    ax.axis("off")
    # Convert plot to an image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    img = Image.open(buf)
    
    return img  # Return the PIL Image

def pad_keypoints(keypoints):
    num_kp = len(keypoints)
    if num_kp > 4:
        keypoints = keypoints[:parse_config.max_kp]
    keypoints_tensor = torch.zeros((4, 2), dtype=torch.float32)  # Default is (0,0)
    if num_kp > 0:
        keypoints_tensor[:num_kp] = torch.tensor(keypoints, dtype=torch.float32)
    return keypoints_tensor

def gaussian_heatmap(height, width, center, sigma=10):
    xs = torch.arange(0, width, dtype=torch.float32, device=center.device)
    ys = torch.arange(0, height, dtype=torch.float32, device=center.device)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')  # shape (height, width)
    
    x0, y0 = center[0], center[1]
    heatmap = torch.exp(-((x_grid - x0)**2 + (y_grid - y0)**2) / (sigma**2))
    return heatmap

def generate_batch_heatmaps(keypoints, height=224, width=224, sigma=10):
    B, num_keypoints, _ = keypoints.shape
    device = keypoints.device

    heatmaps = torch.zeros(B, 1, height, width, dtype=torch.float32, device=device)

    for b in range(B):
        heatmap_single = torch.zeros(height, width, dtype=torch.float32, device=device)
        for k in range(num_keypoints):
            center = keypoints[b, k]  # (2,)
            if 0 < center[0] < width and 0 < center[1] < height:
                g_heatmap = gaussian_heatmap(height, width, center, sigma)
                heatmap_single += g_heatmap
        heatmaps[b, 0] = heatmap_single
    return heatmaps

def hungarian_algorithm_torch(preds: torch.Tensor, targets: torch.Tensor):
    B, N, _ = preds.shape  # Here, N is assumed to be 4
    all_matches = []
    
    for b in range(B):
        cost_matrix = torch.cdist(preds[b].unsqueeze(0), targets[b].unsqueeze(0)).squeeze(0)
        
        best_cost = float('inf')
        best_perm = None
        
        for perm in itertools.permutations(range(N)):
            perm_tensor = torch.tensor(perm, device=cost_matrix.device)
            total_cost = cost_matrix[torch.arange(N, device=cost_matrix.device), perm_tensor].sum()
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_perm = perm
        
        matches = [(i, best_perm[i]) for i in range(N)]
        all_matches.append(matches)
    return all_matches

def hungarian_distance_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, N, _ = preds.shape
    matches_list = hungarian_algorithm_torch(preds, targets)
    
    total_loss = 0.0
    for b in range(B):
        matches = matches_list[b]
        for pred_idx, target_idx in matches:
            total_loss += torch.norm(preds[b, pred_idx] - targets[b, target_idx], p=1)
    return total_loss / B

def permutation_invariant_mse_loss(pred, target):
    B, K, _ = pred.shape
    perms = list(itertools.permutations(range(K)))
    losses = []
    
    for perm in perms:
        target_perm = target[:, list(perm), :]
        loss_sample = torch.mean(torch.sum((pred - target_perm) ** 2, dim=-1), dim=-1)  # [B]
        losses.append(loss_sample)
    losses = torch.stack(losses, dim=1)
    min_loss, _ = torch.min(losses, dim=1)
    return torch.mean(min_loss)

def chamfer_loss(pred, target):
    B, K, _ = pred.shape
    pred_exp = pred.unsqueeze(2)  
    target_exp = target.unsqueeze(1)
    dists = torch.sqrt(torch.sum((pred_exp - target_exp) ** 2, dim=-1) + 1e-6)
    loss_pred_to_target = torch.mean(torch.min(dists, dim=2)[0])
    loss_target_to_pred = torch.mean(torch.min(dists, dim=1)[0])
    
    return loss_pred_to_target + loss_target_to_pred

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def iou_loss(score, target):
    target = target.float()
    smooth = 1e-5
    tp_sum = torch.sum(score * target)
    fp_sum = torch.sum(score * (1 - target))
    fn_sum = torch.sum((1 - score) * target)
    loss = (tp_sum + smooth) / (tp_sum + fp_sum + fn_sum + smooth)
    loss = 1 - loss
    return loss

class CombinedViTKeypointModel(nn.Module):
    def __init__(self, vit_heatmap, keypoint_decoder):
        super().__init__()
        self.vit_heatmap = vit_heatmap  # Sub-model 0 (ViT Encoder + Heatmap)
        self.keypoint_decoder = keypoint_decoder  # Sub-model 1 (Keypoint Decoder)

    def forward(self, x):
        patch_tokens, heatmap = self.vit_heatmap(x)  # Extract features and heatmap
        coords = self.keypoint_decoder(heatmap)  # Predict keypoints
        return heatmap, coords
        
def normalize_images(images):
    min_vals = np.min(images, axis=(1, 2), keepdims=True)
    max_vals = np.max(images, axis=(1, 2), keepdims=True)
    
    ranges = max_vals - min_vals
    
    ranges[ranges == 0] = 1
    
    normalized_images = (images - min_vals) / ranges
    return normalized_images

def calculate_iou_for_batch(heatmaps, pred_heatmaps, threshold=0.1):
    heatmaps = heatmaps.squeeze(1).cpu().detach().numpy()
    # pred_heatmaps = pred_heatmaps.squeeze(1)
    pred_heatmaps = normalize_images(pred_heatmaps)
    heatmaps = normalize_images(heatmaps)
    
    binary_heatmaps = (heatmaps > 0.1).astype(np.uint8)
    binary_pred_heatmaps = (pred_heatmaps > 0.4).astype(np.uint8)
    
    intersection = np.logical_and(binary_heatmaps, binary_pred_heatmaps).sum(axis=(1, 2))
    union = np.logical_or(binary_heatmaps, binary_pred_heatmaps).sum(axis=(1, 2))
    # print("*****", binary_heatmaps.sum(axis=(1, 2)), binary_pred_heatmaps.sum(axis=(1, 2)), intersection, union)
    
    iou_per_image = np.where(union == 0, 1.0, intersection / union)
    
    average_iou = np.sum(iou_per_image)
    return average_iou

def calculate_mse(heatmaps, pred_heatmaps):
    heatmaps = heatmaps.squeeze(1).cpu().detach().numpy()
    mse = np.mean((heatmaps - pred_heatmaps) ** 2)
    return mse
    
def point_mse(pred, actual):
    mse_distance = ((pred[0] - actual[0])**2 + (pred[1] - actual[1])**2)**0.5
    return mse_distance

def calculate_ssim(heatmaps, pred_heatmaps):
    heatmaps = heatmaps.squeeze(1).cpu().detach().numpy()
    batch_size = heatmaps.shape[0]
    ssim_total = 0.0
    
    for i in range(batch_size):
        gt = heatmaps[i, :, :]
        pred = pred_heatmaps[i, :, :]
        
        ssim_val = ssim(gt, pred, data_range=1.0)
        ssim_total += ssim_val
    
    # avg_ssim = ssim_total / batch_size
    return ssim_total


def correct_percentage(model, images, labels, mode, epoch):
    model.eval()
    total_point = 0
    incorrect_point = 0
    total_iou = 0
    total_mse = 0
    total_ssim = 0
    with torch.no_grad():
        for idx in range(0, len(images), parse_config.bs):
            if idx+parse_config.bs > len(images):
                break
            image = images[idx:idx+parse_config.bs] # batch data
            image = np.array(image)
            image_tensor = torch.tensor(image).transpose(1, 3).transpose(2, 3).to(device)
            keypoints = []
            for label_1 in labels[idx:idx+parse_config.bs]:
                keypoints.append(pad_keypoints(label_1))
            keypoints = torch.tensor(np.array(keypoints))
            heatmaps = generate_batch_heatmaps(keypoints, height=224, width=224, sigma=parse_config.heatmap_sigma).to(device)
            pred_heatmaps = model(image_tensor)

            pred_heatmaps = pred_heatmaps.squeeze().cpu().detach().numpy() # (16, 256, 256)
            # original_heatmaps = heatmaps.squeeze().cpu().detach().numpy()
            
            for i, pred in enumerate(pred_heatmaps):
                actuals = np.array([point for point in labels[idx+i] if point != (0, 0)])
                coordinates = heatmap_to_coordinates(pred) #(x, 2)
                coordinates = coordinates[:, [1, 0]] 
                # print(actuals.shape, coordinates.shape)
                coordinates = np.array([(coordinates[i,0], coordinates[i,1]) for i in range(coordinates.shape[0]) if coordinates[i,0] != 0 and coordinates[i,1] != 0])

                total_point += actuals.shape[0]
                if len(coordinates) == 0 or len(actuals) == 0:
                    incorrect_point += len(actuals)
                    continue
                try:
                    matches = hungarian_algorithm(coordinates, actuals)
                    incorrect_matches = [match for match in matches if not check_point_is_correct(coordinates[match[0]], actuals[match[1]], 10)] #parse_config.eu_dist
                    if len(matches) != len(actuals):
                        incorrect_point += abs(len(matches) - len(actuals))
                    for match in matches:
                        total_mse += point_mse(coordinates[match[0]], actuals[match[1]])
                        if not check_point_is_correct(coordinates[match[0]], actuals[match[1]], 10): #parse_config.eu_dist
                            incorrect_point += 1
                            
                    if mode == 'test':        
                        plotted_img = plot_results(image_tensor.cpu().detach().numpy()[i], pred, actuals, coordinates, matches, incorrect_matches)
                        image_path = 'logs_new/{}/image/test_image{}.png'.format(exp_name,i+idx)
                        plotted_img.save(image_path)
                        print(f"Saved: {image_path}")   
                        
                except Exception as e:
                    print("Error occured", e)
            # mse_batch = permutation_invariant_mse_loss(torch.tensor(keypoints).cpu().detach(), torch.tensor(pred_keypoints).cpu().detach())
            # print(heatmaps.shape, pred_heatmaps.shape)
            IoU_batch = calculate_iou_for_batch(heatmaps, pred_heatmaps, threshold=0.5)
            ssim_batch = calculate_ssim(heatmaps, pred_heatmaps)
            
            total_iou += IoU_batch 
            # total_mse += mse_batch
            total_ssim += ssim_batch
        ave_mse = total_mse / total_point
        Correct_percentage = (total_point - incorrect_point) / total_point * 100
        # print("Total Test Image: ", idx)
        ave_iou = total_iou / idx
        ave_ssim = total_ssim / idx

    return Correct_percentage, ave_iou, ave_mse, ave_ssim


def train_model(model, train_dataloader, val_dataloader, loss_mp, loss_co, device, num_epochs=25):
    model = model.to(device)
    model.train()
    best_per = 0 
    best_mse = 10
    min_loss = 1000
    best_ep = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr) #2.1e-5

    
    for epoch in range(num_epochs):
       train_epoch_loss = 0
        train_heatmap_loss = 0
        train_coords_loss = 0
        valid_epoch_loss = 0

        for images, heatmaps, keypoints in train_loader:
            images, heatmaps = images.to(device, dtype=torch.float32), heatmaps.to(device, dtype=torch.float32)
            keypoints = keypoints.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            pred_heatmaps = model(images)
            heatmap_loss = loss_mp(pred_heatmaps, heatmaps) #+ dice_loss(pred_heatmaps, heatmaps)  
            loss = heatmap_loss
            loss.backward()
            optimizer.step()
            train_heatmap_loss += heatmap_loss.item()
            train_coords_loss += heatmap_loss.item()
            train_epoch_loss += loss.item()

        epoch_loss = train_epoch_loss / len(train_dataset)
        epoch_heatmap_loss = train_heatmap_loss / len(train_dataset)
        epoch_coords_loss = train_coords_loss / len(train_dataset)
        print('Epoch [{}/{}], Training Loss: {:.4f} - heatmap Loss: {:.4f}, keypoint loss: {:.4f}'.format(epoch, num_epochs, epoch_loss, epoch_heatmap_loss, epoch_coords_loss))

        model.eval()
        with torch.no_grad():
            for images, heatmaps, keypoints in val_dataloader:
                images, heatmaps = images.to(device, dtype=torch.float32), heatmaps.to(device, dtype=torch.float32)
                keypoints = keypoints.to(device, dtype=torch.float32)
                pred_heatmaps = model(images)
                assert pred_heatmaps.shape == heatmaps.shape, f'pred {pred_heatmaps.shape} and real {heatmaps.shape} heatmap have different shape'
                heatmap_loss = loss_mp(pred_heatmaps, heatmaps)
                # assert pred_keypoints.shape == keypoints.shape, f'pred {pred_keypoints.shape} and real {keypoints.shape} keypoints have different shape'
                # coords_loss = loss_co(pred_keypoints, keypoints)
                loss = heatmap_loss
                valid_epoch_loss += loss.item()
        
        val_epoch_loss = valid_epoch_loss / len(val_dataset)
        # model_cal = heatmap_decoder_model
        Correct_percentage, ave_iou, ave_mse, ave_ssim = correct_percentage(model, val_images, val_labels, 'val', epoch)
        print('              Validation Loss: {:.4f}'.format(val_epoch_loss),
              'Correct_per (points): {:.4f}, Average mse (points): {:.4f}'.format(Correct_percentage, ave_mse))

        if ave_mse < best_mse:
            best_mse = ave_mse
            best_ep = epoch
            torch.save(model.state_dict(), best_path)
            print('Update the best model: epoch {} with mse {:.4f}'.format(epoch, ave_mse))
        else:
            if epoch - best_ep >= parse_config.patience:
                print('Early stopping!')
                break

        torch.save(model.state_dict(), latest_path)
    
    # print('The highest accuracy achieved by U-Net is {:.4f}'.format(best_per))
    print('The best mse achieved by {} is {:.4f}'.format(parse_config.model, best_mse))

def test_model(model, test_images, test_labels, device):
    model.load_state_dict(torch.load(best_path)) # best_path
    model = model.to(device)

    Correct_percentage, ave_iou, ave_mse, ave_ssim = correct_percentage(model, test_images, test_labels,'test', 200)
    print('Test keypoint prediction correct percentage (points): {:.4f}, average mse (points): {:.4f}'.format(Correct_percentage, ave_mse, ))



###################################################################
all_images, all_labels = load_all_data(excel_files, directories)
print("Before augmentation: we have ", len(all_images), " images and ", len(all_labels), " labels.")

if parse_config.gan_enhance == 1:
    gan_image_folder = "./Lan_Data_GAN/epoch_175_restore"
    gan_image_path = [f for f in os.listdir(gan_image_folder) if f.endswith(".png")]
    gan_image_path.sort(key=lambda x: int(x.split('_')[2].split('.')[0][3:]))
    gan_image_paths = [os.path.join(gan_image_folder, path) for path in gan_image_path]
    all_images = [load_gan_image(path) for path in gan_image_paths]
    
all_labels = [[(label[i], label[i+1]) for i in range(0, len(label), 2) if label[i] != 0 and label[i+1] != 0] for label in all_labels]

train_images, test_images, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42)

train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.125, random_state=42) # 0.176 * 0.85 ≈ 0.15

# Apply augmentations
augmented_train_images = []
augmented_train_labels = []
# augmented_train_heatmaps = []
for i in range(20):
    augmented_images, augmented_labels = augment_images_labels(train_images, train_labels, is_train=True)
    augmented_train_images.extend(augmented_images)
    augmented_train_labels.extend(augmented_labels)
    
val_images, val_labels = augment_images_labels(val_images, val_labels, is_train=False)
test_images, test_labels = augment_images_labels(test_images, test_labels, is_train=False)
tmp_images, tmp_labels = augment_images_labels(all_images, all_labels, is_train=False)

train_dataset = KeypointDataset(augmented_train_images, augmented_train_labels, sigma=parse_config.heatmap_sigma)
val_dataset = KeypointDataset(val_images, val_labels, sigma=parse_config.heatmap_sigma)
test_dataset = KeypointDataset(test_images, test_labels, sigma=parse_config.heatmap_sigma)  # No augmentation for test
train_loader = DataLoader(train_dataset, batch_size=parse_config.bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=parse_config.bs, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=parse_config.bs, shuffle=False)
print("After augmentation: ", "Train images:", len(augmented_train_images), "Val images:", len(val_images), "Test images:", len(test_images))

# *****************************************************
# model = ViTHeatmapKeypointModel(pretrained=True, max_keypoints= parse_config.max_kp)
# model = ViTHeatmapModel()
model = UNet()
loss_mp = dice_loss  #dice_loss # nn.SmoothL1Loss() # dice_loss nn.MSELoss() # SmoothL1Loss
loss_co = chamfer_loss # permutation_invariant_mse_loss # chamfer_loss
train_model(model, train_loader, val_loader, loss_mp, loss_co, device=device, num_epochs=parse_config.n_epochs)
test_model(model, test_images, test_labels, device=device) # test_images, test_labels
# tmp_image, tmp_labels = augment_images_labels(all_images, all_labels, is_train=False)
# test_model(model, tmp_image[:35], tmp_labels[:35], device=device)

writer.close()
 