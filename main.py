import argparse
from os.path import isfile as isfile

from style_handler import run_style_transfer, create_noise_input
from data_handler import show_tensor_image, load_image_to_tensor
from VGG19 import load_model


def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--style_image', required=True)
    parser.add_argument('-c', '--content-image', required=True)
    parser.add_argument('-n', '--steps', default=300, type=int)
    parser.add_argument('-b', '--blanck-noise', default=False, type=bool)

    return parser


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    STYLE_IMG, CONTENT_IMG, NUM_STEPS, BLANCK_NOISE = args.style_image, args.content_image, args.steps, args.blanck_noise
    assert isfile(STYLE_IMG), "Style image not found"
    assert isfile(CONTENT_IMG), "Content image not found"

    STYLE_IMG = load_image_to_tensor(STYLE_IMG)
    CONTENT_IMG = load_image_to_tensor(CONTENT_IMG)
    assert STYLE_IMG.size() == CONTENT_IMG.size(), "style and content images do not have the same size"

    cnn = load_model()

    INPUT_IMG = create_noise_input(CONTENT_IMG, cnn.device) if BLANCK_NOISE else CONTENT_IMG.clone()
    show_tensor_image(INPUT_IMG, title='Input Image')

    output = run_style_transfer(cnn, CONTENT_IMG, STYLE_IMG, INPUT_IMG, num_steps=NUM_STEPS)
    show_tensor_image(output, title='Output Image')