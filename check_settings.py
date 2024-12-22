import yaml

with open('/kaggle/working/Stegastamp_pytorch_version/cfg/setting.yaml', 'r') as file:
    settings = yaml.safe_load(file)

KAN_value = settings.get('KAN', None)
pretrained_value = settings.get('pretrained', None)
UNet_value = settings.get('UNet', None)
color_space_value = settings.get('color_space', None)

print("KAN:", KAN_value)
print("pretrained:", pretrained_value)
print("UNet:", UNet_value)
print("color_space:", color_space_value)

print()

print("KAN will only work if UNet is True.\nNOTE : UNet and CMYK doesn't work. Tensor Shape Issue.")