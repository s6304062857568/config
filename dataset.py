import json
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SplitDataset(Dataset):
  def __init__(self, X, y):
      assert len(X) == len(y)
      self.video_list = []

      for index in range(len(X)):
        object = {}
        object['video'] = X[index]
        object['action'] = y[index]
        self.video_list.append(object)
   
  def __len__(self):
      # return length of none-flipped videos in directory
      return len(self.video_list)
 
  def __getitem__(self, idx):
      return self.video_list[idx]

class CustomDataset(Dataset):
    """CustomDataset: a Dataset for Human Action Recognition."""
    def __init__(self, annotation_dict, augmented_dict, video_dir="/content/dataset_videos/examples/", augmented_dir="/content/dataset_videos/augmented-examples/", augment=True, transform=None, poseData=False):
        with open(annotation_dict) as f:
            self.video_list = list(json.load(f).items())

        if augment == True:
            self.augment = augment
            with open(augmented_dict) as f:
                augmented_list = list(json.load(f).items())
            self.augmented_dir = augmented_dir
            # extend with augmented data
            self.video_list.extend(augmented_list)

        self.video_dir = video_dir
        self.poseData = poseData
        self.transform = transform

    def __len__(self):
        # return length of none-flipped videos in directory
        return len(self.video_list)

    def __getitem__(self, idx):
        video_id = self.video_list[idx][0]

        encoding = np.squeeze(np.eye(3)[np.array([0,1,2]).reshape(-1)])
        if self.poseData and self.augment==False:
            joints = np.load(self.video_dir + video_id + ".npy", allow_pickle=True)
            sample = {'video_id': video_id, 'joints': joints, 'action': torch.from_numpy(np.array(encoding[self.video_list[idx][1]])), 'class': self.video_list[idx][1]}
        else:
            video = self.VideoToNumpy(video_id)
            sample = {'video_id': video_id, 'video': torch.from_numpy(video).float(), 'action': torch.from_numpy(np.array(encoding[int(self.video_list[idx][1])])), 'class': int(self.video_list[idx][1])}

        return sample

    def keystoint(self, x):
        return {int(k): v for k, v in x.items()}

    def VideoToNumpy(self, video_id):
        # get video
        video = cv2.VideoCapture(self.video_dir + video_id + ".mp4")
        video_frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        #if video_frames_count > 16:
        #    print(f'video_id: {video_id} has over frame size({video_frames_count})')

        if not video.isOpened():
            video = cv2.VideoCapture(self.augmented_dir + video_id + ".mp4")
        if not video.isOpened():
            raise Exception("Video file not readable")

        video_frames = []
        max_frames = 16
        frame_count = 0

        if video_frames_count > max_frames:
          skip_frames_window = max(int(video_frames_count/max_frames),1) #default skip is 1
          for frame_counter in range(max_frames):
            video.set(cv2.CAP_PROP_POS_FRAMES,frame_counter*skip_frames_window)
            success,frame = video.read()
            if not success:
              break

            frame = np.asarray([frame[..., i] for i in range(frame.shape[-1])]).astype(float)
            video_frames.append(frame)
        else:
          while (video.isOpened()):
              # read video
              success, frame = video.read()
              if not success:
                  break

              frame = np.asarray([frame[..., i] for i in range(frame.shape[-1])]).astype(float)
              video_frames.append(frame)

              frame_count += 1

              # Break the loop if the desired number of frames is reached
              if frame_count == max_frames:
                  break     

        video.release()
        assert len(video_frames) == 16
        return np.transpose(np.asarray(video_frames), (1,0,2,3))

class CustomDatasetJSON(Dataset):
    """CustomDataset: a Dataset for Human Action Recognition."""
    def __init__(self, annotation, augmented, video_dir="/content/dataset_videos/examples/", augmented_dir="/content/dataset_videos/augmented-examples/", augment=None, transform=None, poseData=False):

        self.video_list = list(annotation.items())

        if augment == True:
            self.augment = augment
            augmented_list = list(augmented.items())
            self.augmented_dir = augmented_dir
            # extend with augmented data
            self.video_list.extend(augmented_list)

        self.video_dir = video_dir
        self.poseData = poseData
        self.transform = transform

    def __len__(self):
        # return length of none-flipped videos in directory
        return len(self.video_list)

    def __getitem__(self, idx):
        video_id = self.video_list[idx][0]

        encoding = np.squeeze(np.eye(3)[np.array([0,1,2]).reshape(-1)])
        if self.poseData and self.augment==False:
            joints = np.load(self.video_dir + video_id + ".npy", allow_pickle=True)
            sample = {'video_id': video_id, 'joints': joints, 'action': torch.from_numpy(np.array(encoding[self.video_list[idx][1]])), 'class': self.video_list[idx][1]}
        else:
            video = self.VideoToNumpy(video_id)
            sample = {'video_id': video_id, 'video': torch.from_numpy(video).float(), 'action': torch.from_numpy(np.array(encoding[int(self.video_list[idx][1])])), 'class': int(self.video_list[idx][1])}

        return sample

    def keystoint(self, x):
        return {int(k): v for k, v in x.items()}

    def VideoToNumpy(self, video_id):
        # get video
        video = cv2.VideoCapture(self.video_dir + video_id + ".mp4")
        video_frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        #if video_frames_count > 16:
        #    print(f'video_id: {video_id} has over frame size({video_frames_count})')

        if not video.isOpened():
            video = cv2.VideoCapture(self.augmented_dir + video_id + ".mp4")
        if not video.isOpened():
            raise Exception("Video file not readable")

        video_frames = []
        max_frames = 16
        frame_count = 0

        if video_frames_count > max_frames:
          skip_frames_window = max(int(video_frames_count/max_frames),1) #default skip is 1
          for frame_counter in range(max_frames):
            video.set(cv2.CAP_PROP_POS_FRAMES,frame_counter*skip_frames_window)
            success,frame = video.read()
            if not success:
              break

            frame = np.asarray([frame[..., i] for i in range(frame.shape[-1])]).astype(float)
            video_frames.append(frame)
        else:
          while (video.isOpened()):
              # read video
              success, frame = video.read()
              if not success:
                  break

              frame = np.asarray([frame[..., i] for i in range(frame.shape[-1])]).astype(float)
              video_frames.append(frame)

              frame_count += 1

              # Break the loop if the desired number of frames is reached
              if frame_count == max_frames:
                  break     

        video.release()
        assert len(video_frames) == 16
        return np.transpose(np.asarray(video_frames), (1,0,2,3))

def VideoToTensor(video_id, data_dir="/content/dataset_videos/examples/", output_dir="tensor-dataset/", max_len=None, fps=None, padding_mode=None):
    # open video file
    path = data_dir + video_id + ".mp4"
    cap = cv2.VideoCapture(path)
    assert (cap.isOpened())

    channels = 3

    # calculate sample_factor to reset fps
    sample_factor = 1
    if fps:
        old_fps = cap.get(cv2.CAP_PROP_FPS)  # fps of video
        sample_factor = int(old_fps / fps)
        assert (sample_factor >= 1)

    # init empty output frames (C x L x H x W)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    time_len = None
    if max_len:
        # time length has upper bound
        if padding_mode:
            # padding all video to the same time length
            time_len = max_len
        else:
            # video have variable time length
            time_len = min(int(num_frames / sample_factor), max_len)
    else:
        # time length is unlimited
        time_len = int(num_frames / sample_factor)

    frames = torch.FloatTensor(channels, time_len, height, width)

    for index in range(time_len):
        frame_index = sample_factor * index

        # read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            # successfully read frame
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames[:, index, :, :] = frame.float()
        else:
            # reach the end of the video
            if padding_mode == 'zero':
                # fill the rest frames with 0.0
                frames[:, index:, :, :] = 0
            elif padding_mode == 'last':
                # fill the rest frames with the last frame
                assert (index > 0)
                frames[:, index:, :, :] = frames[:, index - 1, :, :].view(channels, 1, height, width)
            break

    frames /= 255
    cap.release()
    torch.save(frames, output_dir + video_id + ".pt")

def VideoToNumpy(video_id, data_dir="/content/dataset_videos/examples/", output_dir="/content/dataset_videos/examples/"):
    # get video
    video = cv2.VideoCapture(data_dir+video_id+".mp4")

    if not video.isOpened():
        raise NameError("Video file corrupted, or improper video name")

    video_frames = []
    max_frames = 16
    frame_count = 0
    while (video.isOpened()):
        # read video
        success, frame = video.read()

        if not success:
            break

        frame = np.asarray([frame[..., i] for i in range(frame.shape[-1])])
        video_frames.append(frame)

        # Break the loop if the desired number of frames is reached
        if frame_count == max_frames:
            break     

    video.release()
    assert len(video_frames) == 16
    np.save(output_dir+video_id+".npy", np.transpose(np.asarray(video_frames), (1,0,2,3)))

def convertAllVideoTensor(path="/content/dataset_videos/annotation_dict.json", data_dir="/content/dataset_videos/examples/", output_dir="/content/dataset_videos/examples/"):
    # Let's convert all video to .pt files
    with open(path) as f:
        video_list = list(json.load(f).items())

    i = 1
    for video_id in video_list:
        print(video_id[0])
        print("Video: ", i)
        VideoToTensor(video_id[0], data_dir, output_dir, max_len=16, fps=10, padding_mode='last')
        i += 1

def convertAllVideoNumpy(path="/content/dataset_videos/annotation_dict.json", data_dir="/content/dataset_videos/examples/", output_dir="numpy-dataset/"):
    # Let's convert all video to .npy files
    with open(path) as f:
        video_list = list(json.load(f).items())

    i = 1
    for video_id in video_list:
        print(video_id[0])
        print("Video: ", i)
        print(i/37085)
        VideoToNumpy(video_id[0], data_dir, output_dir)
        i += 1

def returnWeights(annotation_dict='/content/dataset_videos/annotation_dict.json', labels_dict='/content/dataset_videos/labels_dict.json'):
    # Read Dictionary from dataset
    with open(annotation_dict) as f:
        annotation_dict = json.load(f)

    def keystoint(x):
        return {int(k): v for k, v in x.items()}

    with open(labels_dict) as f:
        labels_dict = json.load(f, object_hook=keystoint)

    # Let's first visualize the distribution of actions in the
    count_dict = dict()
    for key in annotation_dict:
        if labels_dict[annotation_dict[key]] in count_dict:
            count_dict[labels_dict[annotation_dict[key]]] += 1
        else:
            count_dict[labels_dict[annotation_dict[key]]] = 1

    for key in count_dict:
        count_dict[key] = count_dict[key]/37085

    weights = []
    for key, val in labels_dict.items():
        if val != "discard":
            weights.append(count_dict[val])

    return weights


if __name__ == "__main__":

    custom_dataset = CustomDataset(annotation_dict="/content/dataset_videos/annotation_dict.json",
                                       augmented_dict="/content/dataset_videos/augmented_annotation_dict.json")
              
    print(custom_dataset[2]['video_id'])
    print(custom_dataset[2]['class'])
    print(len(custom_dataset))
