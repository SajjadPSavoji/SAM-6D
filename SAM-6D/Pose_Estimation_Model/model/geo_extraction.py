import torch
import torch.nn as nn
import os

from kpconv import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock, nearest_upsample

class GeoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = KPConvFPN(cfg)

        if cfg.pre_trained:
            checkpoint = os.path.join('checkpoints', 'geotransformer-modelnet.pth.tar')
            if not os.path.isfile(checkpoint):
                if not os.path.isdir('checkpoints'):
                    os.makedirs('checkpoints')
                url = "https://github.com/qinzheng93/GeoTransformer/releases/download/1.0.0/geotransformer-modelnet.pth.tar"
                os.system(f"wget -O {checkpoint} {url}")

            loaded_cp = torch.load(checkpoint, map_location='cpu')
            print("load pre-trained checkpoint from: %s" % checkpoint)
            checkpoint_model = loaded_cp['model']
            breakpoint()
            msg = self.load_state_dict(checkpoint_model, strict=False)

    def forward(self, feats, data_dict):
        return self.backbone(feats, data_dict)


class KPConvFPN(nn.Module):
    def __init__(self, cfg):
        super(KPConvFPN, self).__init__()
        input_dim   = cfg.input_dim
        output_dim  = cfg.output_dim
        init_dim    = cfg.init_dim
        kernel_size = cfg.kernel_size
        init_radius = cfg.base_radius * cfg.init_voxel_size
        init_sigma  = cfg.base_sigma  * cfg.init_voxel_size 
        group_norm  = cfg.group_norm
        pre_trained = cfg.pre_trained

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)

        self.encoder2_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = ResidualBlock(
            init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )

        self.encoder3_1 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm, strided=True
        )
        self.encoder3_2 = ResidualBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )

        self.decoder2 = UnaryBlock(init_dim * 12, init_dim * 4, group_norm)
        self.decoder1 = LastUnaryBlock(init_dim * 6, output_dim)

    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']

        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1])

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2])

        latent_s3 = feats_s3
        feats_list.append(feats_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)

        latent_s1 = nearest_upsample(latent_s2, upsampling_list[0])
        latent_s1 = torch.cat([latent_s1, feats_s1], dim=1)
        latent_s1 = self.decoder1(latent_s1)
        feats_list.append(latent_s1)

        feats_list.reverse()

        return feats_list