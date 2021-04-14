import argparse
from pathlib import Path

import cv2
# from array2gif import write_gif
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import net
from function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


# def style_transfer(vgg, decoder, content, style, alpha=1.0,
#                    interpolation_weights=None):
#     assert (0.0 <= alpha <= 1.0)
#     content_f = vgg(content)
#     style_f = vgg(style)
#     if interpolation_weights:
#         _, C, H, W = content_f.size()
#         feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
#         base_feat = adaptive_instance_normalization(content_f, style_f)
#         for i, w in enumerate(interpolation_weights):
#             feat = feat + w * base_feat[i:i + 1]
#         content_f = content_f[0:1]
#     else:
#         feat = adaptive_instance_normalization(content_f, style_f)
#     feat = feat * alpha + content_f * (1 - alpha)
#     return decoder(feat)


def style_transfer(vgg, decoder, content, style, alpha=1.0, beta=None,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    if beta:
        assert (0.0 <= beta <= 1.0)

    content_f = vgg(content)
    style_f = vgg(style)

    # handling beta
    if beta is not None:
        # style_f_mean = style_f.mean()
        # style_f_std = style_f.std()
        # std_style_f = (style_f - style_f_mean) / style_f_std
        # style_f = std_style_f * style_f_std + beta
        # print(f'beta: {beta}')
        style_f_mean = style_f.mean()
        style_f = style_f - style_f_mean + beta

    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
# parser.add_argument('--save_ext', default='.jpg',
#                     help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Style Variation
# parser.add_argument('--beta', type=float, default=None,
#                     help='the offset that controls the style representation, '
#                          'default is None, which means it is original. '
#                          'Should be between 0. and 1.')

parser.add_argument('--num_frame', type=int, default=10,
                    help='the number of frames in the output video, which contributes to the beta, '
                         'beta = 1 / num_frame * current_frame_no.')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(args.beta)

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

assert args.num_frame > 0

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)

        betas = [1 / args.num_frame * i for i in range(args.num_frame + 1)]
        with torch.no_grad():
            # [1, c, h, w] -> [c, h, w] -> [h, w, c]
            outputs = [
                style_transfer(vgg, decoder, content, style, args.alpha, beta, interpolation_weights).squeeze(0).mul(
                    255).add_(0.5).clamp_(
                    0, 255).permute(1, 2, 0).to('cpu',
                                                torch.uint8).numpy() for beta in betas]

        output_name = output_dir / '{:s}_interpolation.mp4'.format(
            content_path.stem)

        h, w, c = outputs[0].shape
        out = cv2.VideoWriter(str(output_name), cv2.VideoWriter_fourcc(*'mp4v'), args.num_frame, (w, h), True)
        for output in outputs:
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            out.write(output)  # frame is a numpy.ndarray with shape (h, w, c)
        out.release()

    else:  # process one content and one style
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            if args.preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)

            betas = [1 / args.num_frame * i for i in range(args.num_frame + 1)]

            with torch.no_grad():
                # [1, c, h, w] -> [c, h, w] -> [h, w, c]
                outputs = [
                    style_transfer(vgg, decoder, content, style, args.alpha, beta).squeeze(0).mul(255).add_(0.5).clamp_(
                        0, 255).permute(1, 2, 0).to('cpu',
                                                    torch.uint8).numpy() for beta in betas]

            output_name = output_dir / '{:s}_stylized_{:s}.mp4'.format(
                content_path.stem, style_path.stem)

            h, w, c = outputs[0].shape
            out = cv2.VideoWriter(str(output_name), cv2.VideoWriter_fourcc(*'mp4v'), args.num_frame, (w, h), True)
            for output in outputs:
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                out.write(output)  # frame is a numpy.ndarray with shape (h, w, c)
            out.release()
