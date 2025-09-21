def gradcam(model, img_tensor, target_layer="layer4"):
    """
    Grad-CAM cho 1 ảnh đầu vào.
    img_tensor: Tensor có shape (1,3,H,W), chưa cần requires_grad.
    target_layer: tên layer để lấy feature map.
    """
    features = {}
    gradients = {}

    def save_features(module, input, output):
        features["value"] = output

    def save_gradients(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    # Gắn hook
    for name, module in model.named_modules():
        if name == target_layer:
            module.register_forward_hook(save_features)
            module.register_full_backward_hook(save_gradients)  # bản mới của PyTorch

    # Đảm bảo input cần grad
    img_tensor = img_tensor.clone().detach()
    img_tensor.requires_grad_(True)

    # Forward
    output = model(img_tensor)
    pred_class = output.argmax().item()

    # Backward
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    # Lấy gradient và feature
    grad = gradients["value"].detach().cpu().numpy()[0]
    fmap = features["value"].detach().cpu().numpy()[0]

    weights = grad.mean(axis=(1, 2))  # GAP trên gradient
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for w, f in zip(weights, fmap):
        cam += w * f

    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (224, 224))
    if cam.max() != 0:
        cam = cam / cam.max()
    return cam
