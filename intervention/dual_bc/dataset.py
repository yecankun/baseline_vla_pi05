# dataset_simulated.py
import os
import linecache
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Data preprocessing: include ToTensor so transforms work on numpy/PIL images
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data roots and per-path lengths
data_root = 'D:/datasource/piper_data/'
test_data_root = 'D:/datasource/piper_data/'

# default lengths per path (12 paths)
data_len = [[137, 173, 138, 140, 191, 223, 140, 80, 206, 190, 0, 0, 0, 169]]
branch_len, path_len = 1, 14

test_data_len = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 201, 233, 138, 0]]
test_branch_len, test_path_len = 1, 14


def _read_image_as_numpy_rgb(path, size=(224, 224)):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

def _parse_label_line(line):
    if not line:
        print('Warning: empty line encountered when parsing label. Returning None.')
        return None
    s = line.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    # split by comma, allow varying whitespace
    parts = [p.strip() for p in s.split(',') if p.strip() != '']
    if len(parts) == 0:
        return None
    try:
        vals = [float(x) for x in parts]
    except Exception:
        return None
    return vals

class DualDataset(Dataset):
    def __init__(self, data_type, transform=None):
        self.transform = transform if transform is not None else trans
        if data_type == 'train':
            self.data_root = data_root
            self.data_len = data_len
            self.path_len = path_len
            self.branch_len = branch_len
        else:
            self.data_root = test_data_root
            self.data_len = test_data_len
            self.path_len = test_path_len
            self.branch_len = test_branch_len

        # support both flat (per-path) or nested (branch x path) data_len
        def _per_path_len(data_len, i, j):
            if len(data_len) == 0:
                return 0
            if isinstance(data_len[0], (list, tuple)):
                return data_len[i][j]
            return data_len[j]
        self._per_path_len = lambda i, j: _per_path_len(self.data_len, i, j)

        # compute total frames
        total = 0
        for i in range(self.branch_len):
            for j in range(self.path_len):
                total += self._per_path_len(i, j)
        self.total_frames = total

    def __len__(self):
        return self.total_frames

    def _find_location(self, global_idx_1based):
        rem = global_idx_1based
        for i in range(self.branch_len):
            for j in range(self.path_len):
                L = self._per_path_len(i, j)
                if rem <= L:
                    return i + 1, j + 1, rem
                rem -= L
        raise IndexError(f'Global index out of range: {global_idx_1based} (total {self.total_frames})')

    def __getitem__(self, idx):
        # idx is 0-based global frame index
        gidx = idx + 1
        b, p, d = self._find_location(gidx)

        img1_path = os.path.join(self.data_root, f'branch{b}', 'image1', f'path{p}', f'{d}.png')
        val_path = os.path.join(self.data_root, f'branch{b}', 'path', f'pose{p}.txt')

        # read current and next pose (1-based lines)
        curr_line = linecache.getline(val_path, d)
        next_line = linecache.getline(val_path, d + 1)

        curr_pose = _parse_label_line(curr_line)

        if len(curr_pose) != 6:
            raise RuntimeError(f'current pose is not 6 dim: {val_path} line {d}')

        if next_line:
            next_pose = _parse_label_line(next_line)
            if len(next_pose) != 6:
                # if malformed, fallback to current
                next_pose = curr_pose
        else:
            next_pose = curr_pose

        # compute large action (next - current) as tensor
        large_action = torch.tensor(next_pose, dtype=torch.float) - torch.tensor(curr_pose, dtype=torch.float)

        pose = torch.tensor(curr_pose, dtype=torch.float)

        # camera/piper count
        piper_path = os.path.join(self.data_root, f'branch{b}', 'camera1', f'pos{p}.txt')
        piper_label_path = os.path.join(self.data_root, f'branch{b}', 'piper', f'label{p}.txt')
        curr_piper_line = linecache.getline(piper_path, d)

        curr_piper_step = int(curr_piper_line.split()[2])
        curr_piper_step = torch.tensor(curr_piper_step, dtype=torch.float)

        piper_label =int(linecache.getline(piper_label_path, d))
        small_action = torch.tensor(piper_label, dtype=torch.float)

        img1_np = _read_image_as_numpy_rgb(img1_path)
        img1_tensor = self.transform(img1_np)  # [C, H, W]

        return {
            'image': img1_tensor,
            'pose': pose,
            'count': curr_piper_step,
            'task_id': 1,
            'large_action': large_action,
            'small_action': small_action
        }



def get_dataloader(batch_size=4, num_workers=0):
    dataset = DualDataset(data_type='train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def get_test_dataloader(batch_size=4, num_workers=0):
    dataset = DualDataset(data_type='test')
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

if __name__ == '__main__':

    dataloader = get_test_dataloader(batch_size=2, num_workers=0)
    print(len(dataloader.dataset))
    for batch in dataloader:
        print('Batch image shape:', batch['image'].shape)

        print('Batch pose:', batch['pose'])
        print('Batch count:', batch['count'])
        print('Batch task_id:', batch['task_id'])
        print('Batch large_action:', batch['large_action'])
        print('small_action: ',batch['small_action'])
        if batch['small_action'][0] != 0 and batch['small_action'][0] != 1:
            print('small_action unique values:', torch.unique(batch['small_action']))
            print('Warning: small_action has unexpected values. Check data parsing logic.')
            break
