import os
import random
import torch
import torchvision
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from functools import partial
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import transforms

from foolbox_evaluation.utils import get_image, get_imagenet_label_dict, get_root, timeit

model = torchvision.models.resnet18(pretrained=True)
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
imagenet_label_dict = get_imagenet_label_dict()


@timeit
def attack_model(attack, fmodel, images, labels, epsilons):
    return attack(fmodel, images, labels, epsilons=epsilons)


def run_sample_attack():
    # create model
    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
    # transform bounds of model
    fmodel = fmodel.transform_bounds((0, 1))
    # get sample data from imagenet
    # todo: should eval be called here for FGSM or other gradient-based attacks??
    model.eval()
    images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=16)
    # print initial model accuracy on the samples
    print(f"Initial accuracy on samples: {fb.utils.accuracy(fmodel, images, labels)}")

    # Only use images for attack which are correctly classified by the model
    correctly_classified_images_mask = is_classified_correctly(fmodel, images, labels)
    images = images[correctly_classified_images_mask]
    labels = labels[correctly_classified_images_mask]

    # ATTACK
    # attack = fb.attacks.LinfDeepFoolAttack()
    attack = fb.attacks.LinfFastGradientAttack()
    # epsilons = np.linspace(0.0, 0.005, num=3)
    epsilons = [0.01]
    raw, clipped, is_adv = attack_model(attack, fmodel, images, labels, epsilons)
    # raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=epsilons)
    robust_accuracy = 1 - is_adv.type(torch.FloatTensor).mean(axis=-1)

    print("Predictions and robust accuracies: ")
    for i, (eps, acc) in enumerate(zip(epsilons, robust_accuracy)):
        print(f"!!!! Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        adversarials = [j for j, adv in enumerate(is_adv[i]) if adv == True]
        for adv_idx in adversarials:
            img = images[adv_idx]
            adv_img = clipped[i][adv_idx]
            print(f"Ground truth label: '{labels[adv_idx]}'")
            # print(f"Accuracy on original image: "
            #       f"{fb.utils.accuracy(fmodel, img.unsqueeze(0), labels[adv_idx].unsqueeze(0))}")

            original_imagenet_idx = fmodel(img.unsqueeze(0)).argmax().item()
            adv_imagenet_idx = fmodel(adv_img.unsqueeze(0)).argmax().item()
            print(f"Original prediction: {original_imagenet_idx}, "
                  f"{imagenet_label_dict[original_imagenet_idx]}")
            print(f"Adversarial prediction: {adv_imagenet_idx}, "
                  f"{imagenet_label_dict[adv_imagenet_idx]}")
            print(" ")

        first_adv_idx = random.choice(adversarials)
        torchvision.utils.save_image(images[first_adv_idx],
                                     os.path.join(get_root(), f"test_images/{first_adv_idx}_orig_eps_{eps}.jpg"))
        torchvision.utils.save_image(clipped[i][first_adv_idx],
                                     os.path.join(get_root(), f"test_images/{first_adv_idx}_adv_eps_{eps}.jpg"))


def is_classified_correctly(model, images, labels):
    """
    Checks whether the model predicts the "correct", ground-truth labels for the (non-adversarial) input images.
    :param model: foolbox model
    :param images: tensor of non-adversarial images (B, C, W, H)
    :param labels: tensor of ground truth labels for the images (B, N), where N = number of classes
    :return: boolean tensor containing True for images that were correctly classified by the model, and False otherwise
    """
    accuracies = list()
    for image, label in zip(images, labels):
        accuracies.append(fb.utils.accuracy(model, image.unsqueeze(0), label.unsqueeze(0)))
    return torch.tensor(accuracies) != 0


def explain_with_lime(img_p, foolbox_model, random_seed=42):
    img = get_image(img_p)
    explainer = lime_image.LimeImageExplainer()
    transf_img = np.array(get_pil_transform()(img))
    classifier_fn = partial(foolbox_predict, foolbox_model)
    explanation = explainer.explain_instance(transf_img,
                                             classifier_fn,  # classification function
                                             top_labels=5,
                                             hide_color=0,
                                             random_seed=random_seed,
                                             num_samples=1000)  # number of images that will be sent to classification function
    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
    #                                             positive_only=True,
    #                                             num_features=5,
    #                                             hide_rest=False)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=False,
                                                num_features=10,
                                                hide_rest=False)
    print(f"Top labels for image '{img_p}':")
    for idx, label_idx in enumerate(explanation.top_labels, 1):
        print(f"\t{idx})  '{imagenet_label_dict[label_idx]} ({label_idx})'")

    probs = classifier_fn([transf_img])
    print(f"Predicted class: {probs.argmax()}, probability: {probs.max()}")

    img_boundary1 = mark_boundaries(temp / 255.0, mask)
    return img_boundary1


def foolbox_predict(fmodel: fb.PyTorchModel, images: np.array):
    batch = torch.stack(tuple(transforms.ToTensor()(i) for i in images), dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = batch.to(device)
    logits = fmodel(images)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf


if __name__ == "__main__":
    # one pixel attack
    # orig_p = "/home/steffi/dev/master_thesis/adversarial_attacks/foolbox-evaluation/test_images/puppy_224-224_original.jpg"
    # adv_p = "/home/steffi/dev/master_thesis/adversarial_attacks/foolbox-evaluation/test_images/puppy_224-224_adv.jpg"
    # other attack
    orig_p = "/home/steffi/dev/master_thesis/adversarial_attacks/foolbox-evaluation/test_images/0_bull-mastiff_orig_eps_0.01.jpg"
    adv_p = "/home/steffi/dev/master_thesis/adversarial_attacks/foolbox-evaluation/test_images/0_bull-mastiff_adv_eps_0.01.jpg"

    model.eval()

    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
    # transform bounds of model
    fmodel = fmodel.transform_bounds((0, 1))
    # get sample data from imagenet

    orig_boundary = explain_with_lime(orig_p, fmodel)
    adv_boundary = explain_with_lime(adv_p, fmodel)

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(orig_boundary)
    ax[1].imshow(adv_boundary)
    plt.show()

    # run_sample_attack()
