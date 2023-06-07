import glob
from abc import ABC
import pandas as pd
import numpy as np
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger
from utils.utils import unpickle
import torch
import torchaudio.transforms as T

class I3DFeaturesDataset(data.Dataset):
    def __init__(self, path, transform=None):
        dict = unpickle(path)
        self.data = np.array(list(map(lambda entry: entry["features_RGB"], dict["features"])))
        self.labels = np.array(dict["labels"])
        self.transform = transform

    def __len__(self):
        return len(self.labels);
  
    def __getitem__(self, index):
        record = self.data[index]
        label = self.labels[index]

        if self.transform is not None:
            record = self.transform(record)

        return record, label

class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat

        if self.load_feat:
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features",
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

    def _get_train_indices(self, record, modality='RGB'):
        ##################################################################
        # TODO: implement sampling for training mode                     #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        clip_size = np.round(record.num_frames[modality] / self.num_clips) # sample length / number of desired clips
        
        frames = []
        centroids = np.linspace(clip_size/2, record.num_frames[modality] - clip_size/2, self.num_clips).round().tolist()
        
        for cen in centroids:
            if self.dense_sampling[modality]:
                # dense sampling
                first = cen - np.around(self.num_frames_per_clip[modality]/2) * self.stride
                
                aux = np.array([first+j*self.stride for j in range(self.num_frames_per_clip[modality])])
                aux[aux > record.num_frames[modality]] = record.num_frames[modality]
                aux[aux < 0] = 0
                
                frames.append(aux)
            else:
                # uniform sampling
                frames.append(np.linspace(
                    cen-clip_size/2,   # first frame
                    cen+clip_size/2-1, # last frame
                    self.num_frames_per_clip[modality]
                ).round().tolist())
        
        return np.array(frames).flatten()
        #raise NotImplementedError("You should implement _get_train_indices")

    def _get_val_indices(self, record, modality):
        ##################################################################
        # TODO: implement sampling for testing mode                      #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        clip_size = np.round(record.num_frames[modality] / self.num_clips) # sample length / number of desired clips
        
        frames = []
        centroids = np.linspace(clip_size/2, record.num_frames[modality] - clip_size/2, self.num_clips).round().tolist()
        
        for cen in centroids:
            if self.dense_sampling[modality]:
                # dense sampling
                first = cen - np.around(self.num_frames_per_clip[modality]/2) * self.stride
                
                aux = np.array([first+j*self.stride for j in range(self.num_frames_per_clip[modality])])
                aux[aux > record.num_frames[modality]] = record.num_frames[modality]
                aux[aux < 0] = 0
                
                frames.append(aux)
            else:
                # uniform sampling
                frames.append(np.linspace(
                    cen-clip_size/2,   # first frame
                    cen+clip_size/2-1, # last frame
                    self.num_frames_per_clip[modality]
                ).round().tolist())
        
        return np.array(frames).flatten()
        #raise NotImplementedError("You should implement _get_train_indices")

    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]

        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0]
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
            images.extend(frame)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added

            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)
    
class ActioNetDataset(data.Dataset):
    def __init__(self, base_data_path, rgb_path, num_clips, modality, transform=None):
        df = unpickle(base_data_path)

        # extracting dataset info
        self.labels = df['verb_class'].to_numpy()
        self.EMG = df['emg_matrix'].to_numpy()
        self.RGB_data = np.array(list(map(lambda entry: entry["features_RGB"], unpickle(rgb_path)["features"])))
        
        # additional parameters
        segment_size = self.EMG[0].shape[0] # n. of measures in a single segment
        self.clip_starts, self.clip_stops = self.compute_clip_boundaries(segment_size, num_clips)
        
        self.modality = modality  # EMG / RGB / ALL

        self.transform = transform  # dizionario con chiavi 'EMG' e 'RGB'

        # initializing spectrogram object
        n_fft = 32
        win_length = None
        hop_length = 4

        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            normalized=True
        )
    
    def __getitem__(self, idx):
        features_RGB = torch.empty(0)  # I3D features
        features_EMG = torch.empty(0)  # spectrogram
        label = self.labels[idx]

        # dividing the data into clips
        if self.modality == 'RGB' or self.modality == 'ALL':
            features_RGB = torch.Tensor(self.RGB_data[idx])

        if self.modality == 'EMG' or self.modality == 'ALL':
            # return EMG spectrogram
            emg_data = torch.tensor(self.EMG[idx])  # raw EMG samples

            features_EMG = []
            for clip_start, clip_stop in zip(self.clip_starts, self.clip_stops):
                features_EMG.append(self.compute_spectrogram(emg_data[clip_start:clip_stop]))
            features_EMG = torch.Tensor(np.array(features_EMG))

        if self.modality not in ['RGB', 'EMG', 'ALL']:
            raise Exception(f"Modality '{self.modality}' is not supported.")

        # Applying the transformation (if needed)
        if self.transform:
            if self.transform["RGB"]:
                features_RGB = self.transform["RGB"](features_RGB)
            if self.transform["EMG"]:
                features_EMG = self.transform["EMG"](features_EMG)

        return {"RGB": features_RGB, "EMG": features_EMG}, label
        
    def compute_clip_boundaries(self, segment_size, n_clips):
        size = np.around(segment_size / n_clips)

        start_points = np.linspace(0, segment_size - size, n_clips).astype(int)
        end_points = start_points - 1
        end_points = np.roll(end_points, -1)
        end_points[-1] = segment_size - 1

        return start_points, end_points

    def compute_spectrogram(self, signal):
        freq_signal = np.array([np.array(self.spectrogram(signal[:, i])) for i in range(16)])
        return freq_signal

    def __len__(self):
        return len(self.labels)
    
class ActionetDataset_2D(Dataset):
    def __init__(self, base_data_path, rgb_path, num_clips, modality, transform=None):
        df = unpickle(base_data_path)

        # extracting dataset info
        self.labels = df['verb_class'].to_numpy()
        self.EMG = df['emg_matrix'].to_numpy()
        self.RGB_data = np.array(list(map(lambda entry: entry["features_RGB"], unpickle(rgb_path)["features"])))
        
        # additional parameters
        segment_size = self.EMG[0].shape[0] # n. of measures in a single segment
        self.clip_starts, self.clip_stops = self.compute_clip_boundaries(segment_size, num_clips)
        
        self.modality = modality  # EMG / RGB / ALL

        self.transform = transform  # dizionario con chiavi 'EMG' e 'RGB'

        # initializing spectrogram object
        n_fft = 32
        win_length = None
        hop_length = 4

        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            normalized=True
        )
    
    def __getitem__(self, idx):
        features_RGB = torch.empty(0)  # I3D features
        features_EMG = torch.empty(0)  # spectrogram
        label = self.labels[idx]

        # dividing the data into clips
        if self.modality == 'RGB' or self.modality == 'ALL':
            features_RGB = torch.Tensor(self.RGB_data[idx])

        if self.modality == 'EMG' or self.modality == 'ALL':
            # return EMG spectrogram
            emg_data = torch.tensor(self.EMG[idx])  # raw EMG samples
            
            emg_data = torch.hstack([
                torch.sum(torch.abs(emg_data[:,:8]), axis=1).view(-1,1),
                torch.sum(torch.abs(emg_data[:,8:]), axis=1).view(-1,1)
            ])

            features_EMG = []
            for clip_start, clip_stop in zip(self.clip_starts, self.clip_stops):
                features_EMG.append(self.compute_spectrogram(emg_data[clip_start:clip_stop]))
            features_EMG = torch.Tensor(np.array(features_EMG))

        if self.modality not in ['RGB', 'EMG', 'ALL']:
            raise Exception(f"Modality '{self.modality}' is not supported.")

        # Applying the transformation (if needed)
        if self.transform:
            if self.transform["RGB"]:
                features_RGB = self.transform["RGB"](features_RGB)
            if self.transform["EMG"]:
                features_EMG = self.transform["EMG"](features_EMG)

        return {"RGB": features_RGB, "EMG": features_EMG}, label
        
    def compute_clip_boundaries(self, segment_size, n_clips):
        size = np.around(segment_size / n_clips)

        start_points = np.linspace(0, segment_size - size, n_clips).astype(int)
        end_points = start_points - 1
        end_points = np.roll(end_points, -1)
        end_points[-1] = segment_size - 1

        return start_points, end_points

    def compute_spectrogram(self, signal):
        freq_signal = np.array([np.array(self.spectrogram(signal[:, i])) for i in range(2)])
        return freq_signal

    def __len__(self):
        return len(self.labels)