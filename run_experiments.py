import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ------------------ Load CIFAR-10 ------------------ #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# ------------------ Pretrained Model ------------------ #
model = resnet18(pretrained=True)
model.eval()

# ------------------ GradCAM ------------------ #
def gradcam(model, img_tensor, target_layer="layer4"):
    # forward pass
    features = {}
    gradients = {}

    def save_features(module, input, output):
        features["value"] = output

    def save_gradients(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    # register hook
    for name, module in model.named_modules():
        if name == target_layer:
            module.register_forward_hook(save_features)
            module.register_backward_hook(save_gradients)

    output = model(img_tensor)
    pred_class = output.argmax().item()

    # backward pass
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    grad = gradients["value"].detach().cpu().numpy()[0]
    fmap = features["value"].detach().cpu().numpy()[0]

    weights = grad.mean(axis=(1, 2))  # GAP
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for w, f in zip(weights, fmap):
        cam += w * f

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max()
    return cam

# ------------------ Demo 1 ảnh ------------------ #
images, saliency_maps = [], []

for img, label in testloader:
    with torch.no_grad():
        cam = gradcam(model, img)

    images.append(img[0].permute(1,2,0).numpy())  # (H,W,C)
    saliency_maps.append(cam)                     # (H,W)

    if len(images) >= 5:   # lấy 5 ảnh đầu tiên
        break

# ------------------ Visualization ------------------ #
plt.figure(figsize=(10,4))
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(images[i])
    plt.axis("off")

    plt.subplot(2,5,5+i+1)
    plt.imshow(images[i])
    plt.imshow(saliency_maps[i], cmap="jet", alpha=0.5)
    plt.axis("off")
plt.show()
