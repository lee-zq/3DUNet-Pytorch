import numpy as np

def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
            file_name_list.append(lines.split())
    return file_name_list

def padding_img(img, size, stride):
    assert (len(img.shape) == 3)  # 3D array
    img_s, img_h, img_w = img.shape
    leftover_s = (img_s - size) % stride

    if (leftover_s != 0):
        s = img_s + (stride - leftover_s)
    else:
        s = img_s

    tmp_full_imgs = np.zeros((s, img_h, img_w),dtype=np.float32)
    tmp_full_imgs[:img_s] = img
    print("Padded images shape: " + str(tmp_full_imgs.shape))
    return tmp_full_imgs

# Divide all the full_imgs in pacthes
def extract_ordered_overlap(img, size, stride):
    img_s, img_h, img_w = img.shape
    assert (img_s - size) % stride == 0
    N_patches_img = (img_s - size) // stride + 1

    print("Patches number of the image:{}".format(N_patches_img))
    patches = np.empty((N_patches_img, size, img_h, img_w), dtype=np.float32)

    for s in range(N_patches_img):  # loop over the full images
        patch = img[s * stride : s * stride + size]
        patches[s] = patch

    return patches  # array with all the full_imgs divided in patches

def to_one_hot_3d(tensor, n_classes=3):  # shape = [batch, s, h, w]
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot