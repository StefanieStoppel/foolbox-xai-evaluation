import foolbox as fb
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchray.attribution.common import get_module, Probe
from torchray.attribution.grad_cam import gradient_to_grad_cam_saliency, grad_cam

from torchray.attribution.guided_backprop import guided_backprop
from torchray.benchmark import get_example_data, plot_example
from torchray.utils import imsc
from torchvision import transforms

from foolbox_evaluation.docs_example import get_pil_transform
from foolbox_evaluation.utils import get_imagenet_label_dict, get_image


def run_gradcam_default():
    # Obtain example data.
    model, x, category_id, _ = get_example_data()

    # Grad-CAM backprop.
    saliency = grad_cam(model, x, category_id, saliency_layer='features.29')

    # Plots.
    plot_example(x, saliency, 'grad-cam backprop', category_id, show_plot=True)


def run_guided_backprop_default():
    # Obtain example data.
    model, x, category_id, _ = get_example_data()

    # Guided backprop.
    saliency = guided_backprop(model, x, category_id)

    # Plots.
    plot_example(x, saliency, 'guided backprop', category_id, show_plot=True)


def run_guided_backprop(model, image, label_id):
    # Guided backprop.
    saliency = guided_backprop(model, image, label_id)

    # Plot
    plot_example(image, saliency, 'guided backprop', label_id, show_plot=True)


def diff_guided_backprop(model, orig_image, orig_label_id, adv_image, adv_label_id):
    # Guided backprop.
    orig_saliency = guided_backprop(model, orig_image, orig_label_id)
    adv_saliency = guided_backprop(model, adv_image, adv_label_id)
    diff_saliency = torch.abs(orig_saliency - adv_saliency)
    # images = torch.cat((orig_saliency.detach().cpu(), adv_saliency.detach().cpu(), diff_saliency.detach().cpu()), dim=0)

    fig = plt.figure(figsize=(10, 5))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.set_title(f'Original saliency: {imagenet_label_dict[orig_label_id]} ({orig_label_id})')
    imsc(orig_saliency.squeeze(0), interpolation=None)

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.set_title(f'Adv saliency: {imagenet_label_dict[adv_label_id]} ({adv_label_id})')
    imsc(adv_saliency.squeeze(0), interpolation=None)

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.set_title(f'Diff saliency')
    imsc(diff_saliency.squeeze(0), interpolation=None)
    plt.show()


def imshow(img):
    npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray', vmin=0, vmax=1)
    plt.imshow(npimg, cmap='gray', vmin=0, vmax=1)
    plt.show()


if __name__ == "__main__":
    model = torchvision.models.resnet18(pretrained=True)
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

    imagenet_label_dict = get_imagenet_label_dict()

    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

    original_p = "/home/steffi/dev/master_thesis/adversarial_attacks/foolbox-evaluation/test_images/2_orig_eps_0.05.jpg"
    adv_p = "/home/steffi/dev/master_thesis/adversarial_attacks/foolbox-evaluation/test_images/2_adv_eps_0.05.jpg"
    orig_label = 438
    adv_label = 440

    original = transforms.ToTensor()(np.array(get_pil_transform()(get_image(original_p))))
    adv = transforms.ToTensor()(np.array(get_pil_transform()(get_image(adv_p))))

    diff_guided_backprop(model, original.unsqueeze(0).cuda(), orig_label, adv.unsqueeze(0).cuda(), adv_label)
    # run_guided_backprop(model, original.unsqueeze(0).cuda(), orig_label)
    # run_guided_backprop(model, adv.unsqueeze(0).cuda(), adv_label)
    print("hi")

    # run_gradcam_default()



