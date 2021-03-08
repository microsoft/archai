import torch
from torch.functional import Tensor
import torch.nn as nn

from typing import Tuple

from skimage import feature


class VisualFeaturesWithFFNet(nn.Module):
    def __init__(self, feature_len:int, n_classes:int, pixels_per_hog_cell:Tuple[int, int]=(8, 8)):
        super(VisualFeaturesWithFFNet, self).__init__()
        self.feature_len = feature_len
        self.pixels_per_hog_cell = pixels_per_hog_cell

        self.net = nn.Sequential(nn.Linear(feature_len, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, n_classes))

    def _compute_features(self, x:Tensor)->Tensor:
        # compute image features on each image
        num_images = x.shape[0]
        hog_feat_storage = []
        for i in range(num_images):

            img = x[i, :, :, :]
            img = img.permute(1, 2, 0)
            img = img.cpu().numpy()
            
            hog_feat = feature.hog(img, pixels_per_cell=self.pixels_per_hog_cell)
            hog_feat = torch.from_numpy(hog_feat)
            assert self.feature_len == hog_feat.shape[0]
            hog_feat_storage.append(hog_feat)

        all_hog_feats = torch.stack(hog_feat_storage)
        all_hog_feats.requires_grad = False
        all_hog_feats = all_hog_feats.cuda()
        return all_hog_feats    


    def forward(self, x):
        feats = self._compute_features(x)    
        return self.net(feats.float())