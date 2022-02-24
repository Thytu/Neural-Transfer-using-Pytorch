import gradio as gr

from typing import List
from VGG19 import load_model
from data_handler import image_to_tensor, tensor_to_image
from style_handler import run_style_transfer, create_noise_input


def fuse_image(
        content,
        style,
        checkboxes: List[str],
        nb_step: int,
        width: int,
        height: int,
        style_weight: float,
        content_weight: float,
    ):
    """
    Fuse content and style image

    :param content: content image
    :param style: style image
    :param blanck_noise: if True, create a random noise image
    :param nb_step: number of iteration for style transfer
    :param width: width of the target image
    :param height: height of the target image

    :return: fused image
    """

    cpu_only = "CPU Only (for low GPU memory)" in checkboxes
    blanck_noise = "Blanck Noise" in checkboxes

    model = MODEL if not cpu_only else MODEL.cpu()

    width = int(width) if width else None
    height = int(height) if height else None

    content = image_to_tensor(content, width=width, height=height, device=model.device)

    output, _ = run_style_transfer(
        model,
        content_img=content,
        style_img=image_to_tensor(style, width=width, height=height, device=model.device),
        input_img=create_noise_input(content, width=width, height=height, device=model.device) if blanck_noise else content.clone(),
        num_steps=nb_step,
        style_weight=style_weight,
        content_weight=content_weight
    )

    del content

    return tensor_to_image(output)


if __name__ == '__main__':

    MODEL = load_model()

    iface = gr.Interface(fn=fuse_image, inputs=[
        gr.inputs.Image(label="Content image"),
        gr.inputs.Image(label="Style image"),
        gr.inputs.CheckboxGroup(["Blanck Noise", "CPU Only (for low GPU memory)"], default=[False, False], type="value", label=None),
        gr.inputs.Number(default=15, label="Number of step", optional=False),
        gr.inputs.Number(label="Image width", optional=True),
        gr.inputs.Number(label="Image height", optional=True),
        gr.inputs.Number(default=1000000, label="Style weight", optional=False),
        gr.inputs.Number(default=1, label="Content weight", optional=False),
    ], outputs=[
        gr.outputs.Image(label="Resulting image"),
    ],
        layout="horizontal", title="Neural Transfert",
        description="Implementation of the pytorch's neural transfer tutorial: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html",
        theme="huggingface",
    )

    iface.launch(enable_queue=False)
