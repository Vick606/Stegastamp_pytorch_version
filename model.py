import sys

sys.path.append("PerceptualSimilarity\\")
import os
import utils
import torch
import numpy as np
from torch import nn
import torchgeometry
from kornia import color
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
from torchvision import transforms


def convert_to_colorspace(image, color_space):
    """Convert RGB image to specified color space with normalization and clipping"""
    image = torch.clamp(image, 0.0, 1.0)  # Ensure input is within [0,1]
    if color_space == "RGB":
        return image
    elif color_space == "HSI":
        return torch.clamp(color.rgb_to_hsv(image), 0.0, 1.0)
    elif color_space == "CMYK":
        # Convert RGB [0,1] to CMYK
        r, g, b = image[:, 0], image[:, 1], image[:, 2]
        k = 1 - torch.max(image, dim=1)[0]
        c = (1 - r - k) / (1 - k + 1e-8)
        m = (1 - g - k) / (1 - k + 1e-8)
        y = (1 - b - k) / (1 - k + 1e-8)
        cmyk = torch.stack([c, m, y, k], dim=1)
        return torch.clamp(cmyk, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")


def convert_from_colorspace(image, color_space):
    """Convert from specified color space back to RGB with normalization and clipping"""
    if color_space == "RGB":
        return torch.clamp(image, 0.0, 1.0)
    elif color_space == "HSI":
        return torch.clamp(color.hsv_to_rgb(image), 0.0, 1.0)
    elif color_space == "CMYK":
        # Convert CMYK back to RGB [0,1]
        c, m, y, k = image[:, 0], image[:, 1], image[:, 2], image[:, 3]
        r = (1 - c) * (1 - k)
        g = (1 - m) * (1 - k)
        b = (1 - y) * (1 - k)
        rgb = torch.stack([r, g, b], dim=1)
        return torch.clamp(rgb, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")


class Dense(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation="relu",
        kernel_initializer="he_normal",
    ):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == "he_normal":
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == "relu":
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation="relu", strides=1
    ):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2)
        )
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation is not None:
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, color_space="RGB"):  # Added color_space parameter
        super(SpatialTransformerNetwork, self).__init__()
        # Determine input channels based on color space
        input_channels = 4 if color_space == "CMYK" else 3

        self.localization = nn.Sequential(
            Conv2D(
                input_channels, 32, 3, strides=2, activation="relu"
            ),  # Modified input channels
            Conv2D(32, 64, 3, strides=2, activation="relu"),
            Conv2D(64, 128, 3, strides=2, activation="relu"),
            Flatten(),
            Dense(320000, 128, activation="relu"),
            nn.Linear(128, 6),
        )
        self.localization[-1].weight.data.fill_(0)
        self.localization[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
        self.color_space = color_space

    def forward(self, image):
        image_converted = image
        theta = self.localization(image_converted)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        transformed_image = F.grid_sample(image_converted, grid, align_corners=False)

        # transformed_image = convert_from_colorspace(transformed_image, self.color_space)
        return transformed_image


class StegaStampEncoder(nn.Module):
    def __init__(self, color_space="RGB", KAN=False):
        super(StegaStampEncoder, self).__init__()

        self.color_space = color_space
        input_channels = 4 if color_space == "CMYK" else 3

        self.secret_dense = Dense(
            100, 7500, activation="relu", kernel_initializer="he_normal"
        )

        self.conv1 = Conv2D(input_channels + 3, 32, 3, activation="relu")
        self.conv2 = Conv2D(32, 32, 3, activation="relu", strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation="relu", strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation="relu", strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation="relu", strides=2)
        self.up6 = Conv2D(256, 128, 3, activation="relu")
        self.conv6 = Conv2D(256, 128, 3, activation="relu")
        self.up7 = Conv2D(128, 64, 3, activation="relu")
        self.conv7 = Conv2D(128, 64, 3, activation="relu")
        self.up8 = Conv2D(64, 32, 3, activation="relu")
        self.conv8 = Conv2D(64, 32, 3, activation="relu")
        self.up9 = Conv2D(32, 32, 3, activation="relu")
        self.conv9 = Conv2D(input_channels + 3 + 32 + 32, 32, 3, activation="relu")
        self.residual = Conv2D(32, input_channels, 1, activation=None)

    def forward(self, inputs):
        secret, image = inputs
        secret = secret - 0.5

        image_converted = convert_to_colorspace(image, self.color_space)
        image_converted = image_converted - 0.5

        secret = self.secret_dense(secret)
        secret = secret.reshape(-1, 3, 50, 50)
        secret_enlarged = nn.Upsample(scale_factor=(8, 8))(secret)

        inputs = torch.cat([secret_enlarged, image_converted], dim=1)

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)

        residual = convert_from_colorspace(residual, self.color_space)

        return residual


class StegaStampDecoder(nn.Module):
    def __init__(self, color_space="RGB", KAN=False, secret_size=100):
        super(StegaStampDecoder, self).__init__()
        self.secret_size = secret_size
        self.color_space = color_space
        input_channels = 4 if color_space == "CMYK" else 3

        self.stn = SpatialTransformerNetwork(color_space=color_space)
        self.decoder = nn.Sequential(
            Conv2D(input_channels, 32, 3, strides=2, activation="relu"),  # Modified input channels
            Conv2D(32, 32, 3, activation="relu"),
            Conv2D(32, 64, 3, strides=2, activation="relu"),
            Conv2D(64, 64, 3, activation="relu"),
            Conv2D(64, 64, 3, strides=2, activation="relu"),
            Conv2D(64, 128, 3, strides=2, activation="relu"),
            Conv2D(128, 128, 3, strides=2, activation="relu"),
            Flatten(),
            Dense(21632, 512, activation="relu"),
            Dense(512, secret_size, activation=None),
        )

    def forward(self, image):
        image_converted = convert_to_colorspace(image, self.color_space)
        image_converted = image_converted - 0.5
        transformed_image = self.stn(image_converted)
        transformed_image = transformed_image + 0.5

        transformed_image = convert_from_colorspace(transformed_image, self.color_space)

        return torch.sigmoid(self.decoder(transformed_image))


        return convert_to_colorspace(torch.sigmoid(self.decoder(transformed_image)), self.color_space)


class StegaStampEncoderUnet(nn.Module):
    def __init__(self, color_space="RGB", KAN=False, bilinear=False):
        super(StegaStampEncoderUnet, self).__init__()

        if KAN:
            import kan_unet_parts as UNet
        else:
            import unet_parts as UNet

        self.color_space = color_space
        input_channels = 4 if color_space == "CMYK" else 3

        self.secret_dense = Dense(
            100, 7500, activation="relu", kernel_initializer="he_normal"
        )

        self.conv1 = nn.Conv2d(input_channels + 3, input_channels + 3, 3, padding=8)
        self.inc = UNet.DoubleConv(input_channels + 3, 64)
        self.down1 = UNet.Down(64, 128)
        self.down2 = UNet.Down(128, 256)
        self.DoubleConv = UNet.DoubleConv(256, 512)
        factor = 2 if bilinear else 1
        self.up1 = UNet.Up(512, 256 // factor, bilinear)
        self.up2 = UNet.Up(256, 128 // factor, bilinear)
        self.up3 = UNet.Up(128, 64 // factor, bilinear)
        self.outc = UNet.OutConv(64, input_channels)
        self.conv2 = nn.Conv2d(input_channels, 3, 15, padding=0)
        self.sig = nn.Sigmoid()

    def forward(self, inputs):
        secret, image = inputs
        secret = secret - 0.5

        image_converted = convert_to_colorspace(image, self.color_space)
        image_converted = image_converted - 0.5

        secret = self.secret_dense(secret)
        secret = secret.reshape(-1, 3, 50, 50)
        image_converted = nn.functional.interpolate(
            image_converted, scale_factor=(1 / 8, 1 / 8)
        )

        inputs = torch.cat([secret, image_converted], dim=1)
        conv1 = self.conv1(inputs)
        x1 = self.inc(conv1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.DoubleConv(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        x = self.conv2(x)

        secret_enlarged = nn.Upsample(scale_factor=(8, 8))(x)
        secret_enlarged = self.sig(secret_enlarged)
        secret_enlarged = convert_from_colorspace(secret_enlarged, self.color_space)

        return secret_enlarged


class StegaStampDecoderUnet(nn.Module):
    def __init__(self, color_space="RGB", KAN=False, secret_size=100):
        super(StegaStampDecoderUnet, self).__init__()
        self.secret_size = secret_size
        self.color_space = color_space
        input_channels = 4 if color_space == "CMYK" else 3

        self.stn = SpatialTransformerNetwork(color_space=color_space)
        self.decoder = nn.Sequential(
            Conv2D(input_channels, 32, 3, strides=2, activation="relu"),  # Modified input channels
            Conv2D(32, 32, 3, activation="relu"),
            Conv2D(32, 64, 3, strides=2, activation="relu"),
            Conv2D(64, 64, 3, activation="relu"),
            Conv2D(64, 64, 3, strides=2, activation="relu"),
            Conv2D(64, 128, 3, strides=2, activation="relu"),
            Conv2D(128, 128, 3, strides=2, activation="relu"),
            Flatten(),
            Dense(21632, 512, activation="relu"),
            Dense(512, secret_size, activation=None),
        )

    def forward(self, image):
        image_converted = convert_to_colorspace(image, self.color_space)
        image_converted = image_converted - 0.5
        transformed_image = self.stn(image_converted)
        transformed_image = transform_image + 0.5

        transformed_image = convert_from_colorspace(transformed_image, self.color_space)

        return torch.sigmoid(self.decoder(transformed_image))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2D(3, 8, 3, strides=2, activation="relu"),
            Conv2D(8, 16, 3, strides=2, activation="relu"),
            Conv2D(16, 32, 3, strides=2, activation="relu"),
            Conv2D(32, 64, 3, strides=2, activation="relu"),
            Conv2D(64, 1, 3, activation=None),
        )

    def forward(self, image):
        x = image - 0.5
        x = self.model(x)
        output = torch.mean(x)
        return output, x


def transform_net(encoded_image, args, global_step):
    sh = encoded_image.size()
    ramp_fn = lambda ramp: np.min([global_step / ramp, 1.0])

    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    rnd_brightness = utils.get_rnd_brightness_torch(
        rnd_bri, rnd_hue, args.batch_size
    )  # [batch_size, 3, 1, 1]
    jpeg_quality = 100.0 - torch.rand(1)[0] * ramp_fn(args.jpeg_quality_ramp) * (
        100.0 - args.jpeg_quality
    )
    rnd_noise = torch.rand(1)[0] * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

    contrast_low = 1.0 - (1.0 - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1.0 + (args.contrast_high - 1.0) * ramp_fn(args.contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = torch.rand(1)[0] * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

    # blur
    N_blur = 7
    f = utils.random_blur_kernel(
        probs=[0.25, 0.25],
        N_blur=N_blur,
        sigrange_gauss=[1.0, 3.0],
        sigrange_line=[0.25, 1.0],
        wmin_line=3,
    )
    if args.cuda:
        f = f.cuda()
    encoded_image = F.conv2d(encoded_image, f, bias=None, padding=int((N_blur - 1) / 2))

    # noise
    noise = torch.normal(
        mean=0, std=rnd_noise, size=encoded_image.size(), dtype=torch.float32
    )
    if args.cuda:
        noise = noise.cuda()
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # contrast & brightness
    contrast_scale = torch.Tensor(encoded_image.size()[0]).uniform_(
        contrast_params[0], contrast_params[1]
    )
    contrast_scale = contrast_scale.reshape(encoded_image.size()[0], 1, 1, 1)
    if args.cuda:
        contrast_scale = contrast_scale.cuda()
        rnd_brightness = rnd_brightness.cuda()
    encoded_image = encoded_image * contrast_scale
    encoded_image = encoded_image + rnd_brightness
    encoded_image = torch.clamp(encoded_image, 0, 1)

# saturation
sat_weight = torch.FloatTensor([0.3, 0.6, 0.1]).reshape(1, 3, 1, 1)
if args.cuda:
    sat_weight = sat_weight.cuda()
encoded_image_lum = torch.mean(encoded_image * sat_weight, dim=1).unsqueeze_(1)
encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

# jpeg
encoded_image = encoded_image.reshape([-1, 3, 400, 400])
if not args.no_jpeg:
    encoded_image = utils.jpeg_compress_decompress(
        encoded_image, rounding=utils.round_only_at_0, quality=jpeg_quality
    )

return encoded_image


def get_secret_acc(secret_true, secret_pred):
    if "cuda" in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = (
        1.0
        - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy()
        / correct_pred.size()[0]
    )
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()
    return bit_acc, str_acc


def build_model(
    encoder,
    decoder,
    discriminator,
    lpips_fn,
    secret_input,
    image_input,
    l2_edge_gain,
    borders,
    secret_size,
    M,
    loss_scales,
    yuv_scales,
    args,
    global_step,
    writer,
):
    test_transform = transform_net(image_input, args, global_step)

    input_warped = torchgeometry.warp_perspective(
        image_input, M[:, 1, :, :], dsize=(400, 400), flags="bilinear"
    )

    mask_warped = torchgeometry.warp_perspective(
        torch.ones_like(input_warped), M[:, 1, :, :], dsize=(400, 400), flags="bilinear"
    )

    input_warped += (1 - mask_warped) * image_input

    residual_warped = encoder((secret_input, input_warped))
    encoded_warped = residual_warped + input_warped

    residual = torchgeometry.warp_perspective(
        residual_warped, M[:, 0, :, :], dsize=(400, 400), flags="bilinear"
    )
