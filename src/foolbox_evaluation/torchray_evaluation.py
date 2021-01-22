import foolbox as fb
import numpy as np
import torchvision
from torchray.attribution.common import get_module, Probe
from torchray.attribution.grad_cam import gradient_to_grad_cam_saliency, grad_cam

from torchray.attribution.guided_backprop import guided_backprop
from torchray.benchmark import get_example_data, plot_example
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

    run_guided_backprop(model, original.unsqueeze(0).cuda(), orig_label)
    run_guided_backprop(model, adv.unsqueeze(0).cuda(), adv_label)
    print("hi")

    # run_gradcam_default()



