import torch, einops, cv2, glob, os
import numpy as np
from torch.utils.data import Dataset

def count_frames(video_path):
    """Count the number of frames in a video."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def make_labels_dic(txt_path):
    with open(txt_path, 'r') as f:
        labels = [line.split(':') for line in f.read().strip().split('\n')]

    for idx, label in enumerate(labels):
        labels[idx] = [int(frame_num) if frame_num.isdigit() else frame_num for frame_num in label]

    return {str(label[-1]): label[:-1] for label in labels}


def classify_frame(vid_num, frame_num, labels_dic):
    label = labels_dic[vid_num]
    default_return = 0  # Set the default return value
    if int(frame_num) < int(label[0]):
        return default_return

    # Iterate over the label in pairs (start, stop)
    for i in range(0, len(label), 2):
        start = int(label[i])
        try:
            stop = int(label[i+2])
        except IndexError:
            stop = float('inf')  # Handle case where no end label is present

        if start <= frame_num <= stop:
            return 1 if i % 4 == 0 else 0

    return default_return

def remove_z_pad(frame, n_channels):
    x = frame.shape[0]
    y = frame.shape[1]
    dpadim = frame
    idx = []
    if n_channels == 3:
        #get rid of x padding
        for i in range(x):
            row = frame[i,:,:]
            if(np.average(row)==0):
                idx.append(i)
        dpadim = np.delete(dpadim,idx,0)
        
        idx = []
        #get rid of y padding
        for j in range(y):
            col = frame[:,j,:]
            if(np.average(col)==0):
                idx.append(j)
        
        dpadim = np.delete(dpadim,idx,1)
    else:
        # get rid of x padding
        for i in range(x):
            row = frame[i, :]
            if np.average(row) == 0:
                idx.append(i)
        dpadim = np.delete(dpadim, idx, 0)

        idx = []
        # get rid of y padding
        for j in range(y):
            col = frame[:, j]
            if np.average(col) == 0:
                idx.append(j)
        dpadim = np.delete(dpadim, idx, 1)

    #delete row ignore by range function
    x2 = dpadim.shape[0]
    if((x-x2)>1):
        dpadim = np.delete(dpadim,0,0)
    
    return dpadim

def get_frame_sequence(video_path, frame_number, depth, size, n_channels, mean, std, interval):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    #print(interval)
    if frame_number < depth:
        rpt_first = depth - frame_number -1    #Number of times to repeat first frame
        frames_to_capture = [i for i in range(frame_number+1)]
        frames_to_capture = [0] * rpt_first + frames_to_capture
        
    elif frame_number < depth * interval:
        while depth * interval > frame_number:
            interval -= 1
            
        # Calculate the frames to capture based on the interval, interval of 1 is every frame
        frames_to_capture = [max(frame_number - i*interval, 0) for i in range(depth)]
        frames_to_capture = frames_to_capture[::-1]  # reverse to capture from the oldest to the newest
        
    else: 
        # Calculate the frames to capture based on the interval, interval of 1 is every frame
        frames_to_capture = [max(frame_number - i*interval, 0) for i in range(depth)]
        frames_to_capture = frames_to_capture[::-1]
    #print(interval)
    #print(frames_to_capture)
    # Read frames from start_frame to frame_number
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count in frames_to_capture:
            rpt = frames_to_capture.count(count)
            
            if n_channels == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frame = remove_z_pad(frame, n_channels)
            frame = cv2.resize(frame, (size, size))
            
            # normalization
            frame = frame / 255.0  # Scale pixel values to [0, 1]
            frame = (frame - mean) / std  # Normalize with mean and std
            for _ in range(rpt):
                frames.append(frame)
        if count > frames_to_capture[-1]:
            break
        
        count += 1
    
    # Release the video capture object
    cap.release()
    
    # Check if the number of frames is less than the required depth
    if len(frames) < depth:
        # Create a zero frame with the same dimensions
        if n_channels == 3:
            zero_frame = np.zeros((size, size, n_channels), dtype=np.float32)
        else:
            zero_frame = np.zeros((size, size), dtype=np.float32)
        
        # Normalize the zero frame
        zero_frame = (zero_frame - mean) / std
        
        # Prepend zero frames until the length of frames equals depth
        frames = [zero_frame] * (depth - len(frames)) + frames
    
    if n_channels == 3:
        frames = einops.rearrange(frames, 'f h w c -> f h w c')     #Turns list into numpy array
    else: 
        frames = einops.rearrange(frames, 'f h w -> f h w')     #Turns list into numpy array
    frames_tensor = torch.from_numpy(frames)

    return frames_tensor

class Custom_Traffic_Dataset(Dataset):
    def __init__(self, depth, size, n_channels, vids_path, labels_path, mean, std, interval):
        self.depth = depth
        self.size = size
        self.n_channels = n_channels
        self.vids_path = vids_path
        self.num_vids =  len(glob.glob(os.path.join(self.vids_path, '*.mp4')))
        self.labels_dic = make_labels_dic(labels_path)
        self.seg_dic = self.make_seg_dic()
        self.mean = mean
        self.std = std
        self.interval = interval
        
    def make_seg_dic(self):
        seg_dic = {}
        for i in range(self.num_vids):
            video_path = os.path.join(self.vids_path, f'{i}.mp4')
            frames = count_frames(video_path)
            seg_dic[str(i)] = []
            for frame_num in range(frames):
                frame_class = classify_frame(str(i), frame_num, self.labels_dic)
                if frame_class == 0 or (frame_class == 1 and frame_num % 2 == 0):  # Skipping logic for class 1
                    seg_dic[str(i)].append(frame_num)
        return seg_dic
        
    def __len__(self):
        return sum(len(self.seg_dic[str(i)]) for i in range(self.num_vids))
    
    def __getitem__(self, idx):
        cumulative_segments = 0
        for video_index in range(self.num_vids):
            num_segments = len(self.seg_dic[str(video_index)])
            if cumulative_segments + num_segments > idx:
                local_idx = idx - cumulative_segments
                frame_num = self.seg_dic[str(video_index)][local_idx]
                frame_class = classify_frame(str(video_index), frame_num, self.labels_dic)
                frame_class = torch.tensor(frame_class)
                vid_path = os.path.join(self.vids_path, f'{video_index}.mp4')
                vid_tensor = get_frame_sequence(vid_path, frame_num, self.depth, self.size, self.n_channels, self.mean, self.std, self.interval)
                return vid_tensor.float(), frame_class
            cumulative_segments += num_segments
        raise IndexError('Global Idx out of range')
    