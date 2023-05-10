from tqdm import tqdm
from unetmodel import UNET
import torch
import torch.nn as nn
import torch.optim 

from utils import (
	load_checkpoint,
	save_checkpoint,
	get_loaders,
	check_accuracy,
	save_predictions_as_imgs,
	)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = '/Users/yerassyl/Documents/Thesis/dataset/train_img'
TRAIN_MASK_DIR = '/Users/yerassyl/Documents/Thesis/dataset/train_masks'
VAL_IMG_DIR = '/Users/yerassyl/Documents/Thesis/dataset/val_imgs'
VAL_MASK_DIR = '/Users/yerassyl/Documents/Thesis/dataset/val_masks'


def train_fn(loader, model, optimizer, loss_fn, scaler):
	loop = tqdm(loader)
	
	for batch_idx, (data, targets) in enumerate(loop):
		data = data.to(device=DEVICE)
		targets = targets.float().to(device=DEVICE)

		# forward
		with torch.cuda.amp.autocast():
			predictions = model(data)
			loss = loss_fn(predictions, targets)

		# backward
		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		# update tqdm loop
		loop.set_postfix(loss=loss.item())

def main():
	model = UNET().to(DEVICE)
	loss_fn = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	train_loader, val_loader = get_loaders(
		TRAIN_IMG_DIR,
		TRAIN_MASK_DIR,
		VAL_IMG_DIR,
		VAL_MASK_DIR,
		BATCH_SIZE,
		NUM_WORKERS,
		PIN_MEMORY,
	)

	scaler = torch.cuda.amp.GradScaler()
	for epoch in range(NUM_EPOCHS):
		train_fn(train_loader, model, optimizer, loss_fn, scaler)

		checkpoint = {
			"state_dict":model.state_dict(),
			"optimizer":optimizer.state_dict(),
		}
		save_checkpoint(checkpoint)

		check_accuracy(val_loader, model, device=DEVICE)

		save_predictions_as_img(val_loader, model, folder="saved_images/", device=DEVICE)

if __name__ == "__main__":
	main()