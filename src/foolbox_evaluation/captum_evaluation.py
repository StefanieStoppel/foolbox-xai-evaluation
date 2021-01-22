import foolbox as fb
import numpy as np
import torchvision

from captum.attr import IntegratedGradients, DeepLift, GuidedBackprop, GuidedGradCam
from captum.attr import visualization as viz
from torchvision import transforms

from foolbox_evaluation.docs_example import get_pil_transform
from foolbox_evaluation.utils import get_imagenet_label_dict, get_image


def attribute_image_features(model, algorithm, input, label, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=label,
                                              **kwargs)
    return tensor_attributions


def run_integrated_gradients(model, image, label_id):
    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(model, ig, image, label_id,
                                              baselines=image * 0, return_convergence_delta=True)
    print('Approximation delta: ', abs(delta))
    visualize_attr(image, attr_ig, label_id, title="Overlayed Integrated Gradients")


# todo: does not work
def run_deeplift(model, image, label_id):
    dl = DeepLift(model)
    attr_dl = attribute_image_features(model, dl, image, label_id, baselines=image * 0)
    visualize_attr(image, attr_dl, label_id, title="Overlayed DeepLift")


def run_guided_backprop(model, image, label_id):
    gb = GuidedBackprop(model)
    attr_gb = attribute_image_features(model, gb, image, label_id)
    visualize_attr(image, attr_gb, label_id, title="Overlayed Guided Backprop")


def run_guided_gradcam(model, layer, image, label_id):
    gb = GuidedGradCam(model, layer)
    attr_gb = attribute_image_features(model, gb, image, label_id)
    visualize_attr(image, attr_gb, label_id, title="Overlayed Guided GradCAM")


def visualize_attr(image, attr, label_id, title):
    attr = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
    image = np.transpose(image.squeeze().cpu().detach().numpy(), (1, 2, 0))

    print('Original Image')
    print('Predicted class:', imagenet_label_dict[label_id])

    _ = viz.visualize_image_attr(np.empty(0), image,
                                 method="original_image", title="Original Image")
    _ = viz.visualize_image_attr(attr, image, method="blended_heat_map", sign="all",
                                 show_colorbar=True, title=title)


if __name__ == "__main__":
    model = torchvision.models.resnet18(pretrained=True)
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

    imagenet_label_dict = get_imagenet_label_dict()

    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

    original_p = "/home/steffi/dev/master_thesis/adversarial_attacks/foolbox-evaluation/test_images/3_orig_eps_0.01.jpg"
    adv_p = "/home/steffi/dev/master_thesis/adversarial_attacks/foolbox-evaluation/test_images/3_adv_eps_0.01.jpg"
    orig_label = 438
    adv_label = 440

    original = transforms.ToTensor()(np.array(get_pil_transform()(get_image(original_p)))).unsqueeze(0).cuda()
    adv = transforms.ToTensor()(np.array(get_pil_transform()(get_image(adv_p)))).unsqueeze(0).cuda()

    # run_integrated_gradients(model, adv, adv_label)
    # run_guided_backprop(model, original, orig_label)
    # run_guided_backprop(model, adv, adv_label)

    # Guided GradCAM
    run_guided_gradcam(model, model.layer4[1].conv2, original, orig_label)
    run_guided_gradcam(model, model.layer4[1].conv2, adv, adv_label)


    print("hi")