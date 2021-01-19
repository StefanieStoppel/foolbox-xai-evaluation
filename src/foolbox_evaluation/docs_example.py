import torch
import torchvision
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt


def run_sample_attack():
    # create model
    model = torchvision.models.resnet18(pretrained=True)
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
    # transform bounds of model
    fmodel = fmodel.transform_bounds((0, 1))
    # get sample data from imagenet
    images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=16)
    # print initial model accuracy on the samples
    print(f"Initial accuracy on samples: {fb.utils.accuracy(fmodel, images, labels)}")
    # ATTACK
    attack = fb.attacks.LinfDeepFoolAttack()
    epsilons = np.linspace(0.0, 0.005, num=2)
    raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=epsilons)
    robust_accuracy = 1 - is_adv.type(torch.FloatTensor).mean(axis=-1)

    print("robust accuracy for perturbations with")
    for i, (eps, acc) in enumerate(zip(epsilons, robust_accuracy)):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        first_adv_idx = next(j for j, adv in enumerate(is_adv[i]) if adv == True)
        original = images[first_adv_idx].permute(1, 2, 0).cpu().numpy()
        adv_img_clipped = clipped[0][first_adv_idx].permute(1, 2, 0).cpu().numpy()

        # display original and adversarial images side by side
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(original)
        ax[1].imshow(adv_img_clipped)
        plt.show()

    plt.plot(epsilons, robust_accuracy.numpy())
    plt.show()
