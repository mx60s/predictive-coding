import torch
import torchmetrics
import numpy as np
import pickle
from torch.utils.data import Dataset
import os
from utils.data_processing import map_files_to_chunks


class AutoencoderDataset(Dataset):
    def __init__(self, source_directory, transform=None, num_samples=-1):
        self.sorted_frames = []
        frames_list = []

        frames_files = sorted([f for f in os.listdir(source_directory) if f.startswith('frames_')])
        i = 0
        for f_file in frames_files:
            frames = np.load(os.path.join(source_directory, f_file))
            frames_list.append(frames)
            i += len(frames)
            if num_samples > 0 and i >= num_samples:
                break

        self.sorted_frames = torch.stack([transform(frame) for stack in frames_list for frame in stack])
        self.num_samples = num_samples if num_samples > 0 else len(self.sorted_frames)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sorted_frames[idx]

class SequentialFrameDataset(Dataset):
    def __init__(self, source_directory, transform=None, seq_len=7, num_samples=-1):
        self.frames_list = []
        self.seq_len = seq_len
        self.transform = transform

        frames_files = sorted([f for f in os.listdir(source_directory) if f.startswith('frames_')])
        for f_file in frames_files:
            frames = np.load(os.path.join(source_directory, f_file))
            if len(frames) >= seq_len + 1:  # Only consider files with enough frames
                if self.transform:
                    frames = [self.transform(frame) for frame in frames]
                self.frames_list.append(torch.stack(frames))

        self.indices = [(i, j) for i, frames in enumerate(self.frames_list) for j in range(len(frames) - seq_len)]
        self.num_samples = num_samples if num_samples > 0 else len(self.sorted_frames)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        file_idx, frame_idx = self.indices[idx]
        frames = self.frames_list[file_idx]
        sequence = frames[frame_idx:frame_idx + self.seq_len]
        pred = frames[frame_idx + self.seq_len]

        return [sequence, pred]

class HeadingDataset(Dataset):
    def __init__(self, source_directory, transform=None, seq_len=7, num_samples=-1):
        self.frames_list = []
        self.headings_list = []
        self.seq_len = seq_len
        self.transform = transform

        # Load frame files
        frames_files = sorted([f for f in os.listdir(source_directory) if f.startswith('frames_')])
        
        # Load coordinates files and calculate headings
        coords_files = sorted([f for f in os.listdir(source_directory) if f.startswith('coords_')])

        for frames_file, coords_file in zip(frames_files, coords_files):
            frames = np.load(os.path.join(source_directory, frames_file))
            coords = np.load(os.path.join(source_directory, coords_file))

            i=0
            if len(frames) >= seq_len + 1:  # Only consider files with enough frames
                if self.transform:
                    frames = [self.transform(frame) for frame in frames]
                self.frames_list.append(torch.stack(frames))

                # Calculate heading changes
                headings = [coord[2] for coord in coords]
                heading_diffs = [self.calculate_heading_diff(headings[i], headings[i-1]) for i in range(1, len(headings))]
                self.headings_list.append(torch.FloatTensor(heading_diffs))

                i += len(frames)
                if num_samples > 0 and i >= num_samples:
                    break

        self.indices = [(i, j) for i, frames in enumerate(self.frames_list) for j in range(len(frames) - seq_len)]
        self.num_samples = min(num_samples, len(self.indices)) if num_samples > 0 else len(self.indices)

    def calculate_heading_diff(self, heading1, heading2):
        diff = heading1 - heading2
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return diff

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        file_idx, frame_idx = self.indices[idx]
        frames = self.frames_list[file_idx]
        headings = self.headings_list[file_idx]

        sequence = frames[frame_idx:frame_idx + self.seq_len]
        pred = frames[frame_idx + self.seq_len]
        heading_sequence = headings[frame_idx:frame_idx + self.seq_len]

        return [(sequence, heading_sequence), pred]
        

class HeadingPredCoordDataset(Dataset):
    def __init__(self, source_directory, transform=None, seq_len=7, num_samples=-1):
        self.frames_list = []
        self.headings_list = []
        self.coords_list = []  # Store full coordinates
        self.seq_len = seq_len
        self.transform = transform

        # Load frame files
        frames_files = sorted([f for f in os.listdir(source_directory) if f.startswith('frames_')])
        
        # Load coordinates files and calculate headings
        coords_files = sorted([f for f in os.listdir(source_directory) if f.startswith('coords_')])

        for frames_file, coords_file in zip(frames_files, coords_files):
            frames = np.load(os.path.join(source_directory, frames_file))
            coords = np.load(os.path.join(source_directory, coords_file))
            
            if len(frames) >= seq_len + 1:  # Only consider files with enough frames
                if self.transform:
                    frames = [self.transform(frame) for frame in frames]
                self.frames_list.append(torch.stack(frames))
                self.coords_list.append(torch.FloatTensor(coords))  # Store full coordinates

                # Calculate heading changes
                headings = [coord[2] for coord in coords]
                heading_diffs = [self.calculate_heading_diff(headings[i], headings[i-1]) for i in range(1, len(headings))]
                self.headings_list.append(torch.FloatTensor(heading_diffs))

        self.indices = [(i, j) for i, frames in enumerate(self.frames_list) for j in range(len(frames) - seq_len)]
        self.num_samples = num_samples

    def calculate_heading_diff(self, heading1, heading2):
        diff = heading1 - heading2
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        return diff

    def __len__(self):
        return len(self.indices) if self.num_samples < 0 else self.num_samples

    def __getitem__(self, idx):
        file_idx, frame_idx = self.indices[idx]
        frames = self.frames_list[file_idx]
        headings = self.headings_list[file_idx]
        coords = self.coords_list[file_idx]

        sequence = frames[frame_idx:frame_idx + self.seq_len]
        coord_pred = coords[frame_idx + self.seq_len]  # Full coordinate array as prediction
        heading_sequence = headings[frame_idx:frame_idx + self.seq_len]

        return [(sequence, heading_sequence), coord_pred]

# fix later
class PredCoordDataset(HeadingPredCoordDataset):
    def __init__(self, source_directory, transform=None, seq_len=7, num_samples=-1):
        self.frames_list = []
        self.coords_list = []  # Store full coordinates
        self.seq_len = seq_len
        self.transform = transform
        self.num_samples = num_samples

        # Load frame files
        frames_files = sorted([f for f in os.listdir(source_directory) if f.startswith('frames_')])
        
        # Load coordinates files and calculate headings
        coords_files = sorted([f for f in os.listdir(source_directory) if f.startswith('coords_')])

        for frames_file, coords_file in zip(frames_files, coords_files):
            frames = np.load(os.path.join(source_directory, frames_file))
            coords = np.load(os.path.join(source_directory, coords_file))
            
            if len(frames) >= seq_len + 1:  # Only consider files with enough frames
                if self.transform:
                    frames = [self.transform(frame) for frame in frames]
                self.frames_list.append(torch.stack(frames))
                self.coords_list.append(torch.FloatTensor(coords))  # Store full coordinates

        self.indices = [(i, j) for i, frames in enumerate(self.frames_list) for j in range(len(frames) - seq_len)]

    def __len__(self):
        return len(self.indices) if self.num_samples < 0 else self.num_samples

    def __getitem__(self, idx):
        file_idx, frame_idx = self.indices[idx]
        frames = self.frames_list[file_idx]
        coords = self.coords_list[file_idx]

        sequence = frames[frame_idx:frame_idx + self.seq_len]
        coord_pred = coords[frame_idx + self.seq_len]  # Full coordinate array as prediction

        return [sequence, coord_pred]
    

class CoordinateDataset(AutoencoderDataset):
    def __init__(self, source_directory, transform=None, num_samples=-1):
        super().__init__(source_directory, transform, num_samples)
        
        self.sorted_coords = []

        # Collect the coordinates info associated with each frame
        coords_list = []
        coords_files = sorted([f for f in os.listdir(source_directory) if f.startswith('coords_')])
        print(coords_files)

        i = 0
        for c_file in coords_files:
            coords = np.load(os.path.join(source_directory, c_file)).astype(np.float32)
            coords_list.append(coords)
            i += len(coords)
            if num_samples > 0 and i >= num_samples:
                break

        self.sorted_coords = [torch.from_numpy(coord) for stack in coords_list for coord in stack]
        self.num_samples = num_samples if num_samples > 0 else len(self.sorted_coords)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return [self.sorted_frames[idx], self.sorted_coords[idx]]

# Functions for training

def train_heading(dataloader, model, loss_fn, optimizer, device) -> float:
    size = len(dataloader.dataset)
    train_loss = 0.0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        x_img, x_head = X
        x_img, x_head, y = x_img.to(device), x_head.to(device), y.to(device)
        optimizer.zero_grad()
        
        #print(X.shape)
        #gen, weights = model(X)
        gen = model(x_img, x_head)
        
        loss = loss_fn(gen, y) 
        loss.backward()
        optimizer.step()

        # Append lists
        train_loss += loss.item()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss/len(dataloader)


def test_heading(dataloader, model, loss_fn, device) -> float:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            x_img, x_head = X
            x_img, x_head, y = x_img.to(device), x_head.to(device), y.to(device)
            #gen, weights = model(X)
            gen = model(x_img, x_head)
            test_loss += loss_fn(gen, y).item()

    test_loss /= num_batches

    print(
        f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return test_loss

def train(dataloader, model, loss_fn, optimizer, device) -> float:
    size = len(dataloader.dataset)
    train_loss = 0.0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        
        #print(X.shape)
        #gen, weights = model(X)
        gen = model(X)

        loss = loss_fn(gen, y) 
        loss.backward()
        optimizer.step()

        # Append lists
        train_loss += loss.item()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss/len(dataloader)


def test(dataloader, model, loss_fn, device) -> float:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            #gen, weights = model(X)
            gen = model(X)
            test_loss += loss_fn(gen, y).item()

    test_loss /= num_batches

    print(
        f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return test_loss

def train_no_pred(dataloader, model, loss_fn, optimizer, device) -> float:
    size = len(dataloader.dataset)
    train_loss = 0.0

    model.train()
    for batch, X in enumerate(dataloader):
        X = X.to(device)
        optimizer.zero_grad()
        
        gen = model(X)

        loss = loss_fn(gen, X) 
        loss.backward()
        optimizer.step()

        # Append lists
        train_loss += loss.item()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss/len(dataloader)


def test_no_pred(dataloader, model, loss_fn, device) -> float:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.0

    model.eval()
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            gen = model(X)
            test_loss += loss_fn(gen, X).item()

    test_loss /= num_batches

    print(
        f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return test_loss