import math
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


def read_image() :
    print("enter image path\n>>", end=' ')
    image_path = input()
    if not os.path.exists(image_path):
        print("Image path not found.")
        return
    
    img = Image.open(image_path).convert("L")
    img_array = np.array(img, dtype=np.int32)
    img_h, img_w = img_array.shape
    
    return img_array, img_h, img_w


def adaptive_predictor_helper(A, B, C) :
    if B <= min(A, C):
        return max(A, C)

    elif B >= max(A, C):
        return min(A, C)

    else:
        return A + C - B

def adaptive_predictor(img, H, W) :
    predicted = np.zeros_like(img)
    error = np.zeros_like(img)

    predicted[0, :] = img[0, :]
    predicted[:, 0] = img[:, 0]
    error[0, :] = 0
    error[:, 0] = 0

    for i in range(1, H):
        for j in range(1, W):
            A = img[i, j - 1]
            B = img[i - 1, j - 1]
            C = img[i - 1, j]

            pr = adaptive_predictor_helper(A, B, C)
            predicted[i, j] = pr
            error[i, j] = img[i, j] - pr
    
    return predicted, error


""" def Uniform_Quantizer(Array, N_bits = 2) :  # not like lecture 
    levels = 2**N_bits
    array_min = np.min(Array)
    array_max = np.max(Array)
    step_size = math.ceil((array_max - array_min) / levels)
    
    indexs = np.floor((Array - array_min) / step_size).astype(int)
    indexs = np.clip(indexs, 0, levels - 1)
    
    boundaries = array_min + np.arange(levels + 1) * step_size
    Q_bar = ((boundaries[:-1] + boundaries[1:]) / 2).astype(np.int32)
        
    return indexs, Q_bar, step_size """


# just for showing images
def show_images(images, titles): 
    plt.figure(figsize=(15, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def Compress() :
    # read image 
    img_array, img_h, img_w = read_image()
    
    # predictor
    predicted, error = adaptive_predictor(img_array, img_h, img_w)
    
    # get Uniform quantizer bits
    while True :
        bits = input("Number of quantizer bits (Default=2): ")
        if bits.isdigit() :
            bits = int(bits)
            break
        else :
            print("# enter valid integer.")
    # Uniform quantizer
    indexes, Q_bar = Uniform_Quantizer(error ,bits)
    
    # Compression ration
    original_bits = img_h * img_w * 8
    compressed_bits = img_h * img_w * bits
    ratio = original_bits / compressed_bits
    
    print(f"Compression ratio = {ratio:.3f} \n")
    
    # show images
    show_images(
                [img_array, predicted, error, indexes],
                ["Original", "Predicted", "Error", "Quantized"]
            )
    
    # save compressed files
    np.save("indexes.npy", indexes)
    
    np.save("qbar.npy", Q_bar)
    
    edges = np.zeros((img_h, img_w), dtype=np.int32)
    edges[0, :] = img_array[0, :]
    edges[:, 0] = img_array[:, 0]
    np.save("edges.npy", edges)



def Uniform_DeQuantizer(indexes, Q_bar):
    return Q_bar[indexes]


def Reconstruct_Image(dq, img_edges):
    H, W = dq.shape
    rec = np.zeros((H, W), dtype=np.int32)

    rec[0, :] = img_edges[0, :]
    rec[:, 0] = img_edges[:, 0]

    for i in range(1, H):
        for j in range(1, W):
            A = rec[i, j-1]
            B = rec[i-1, j-1]
            C = rec[i-1, j]

            pr = adaptive_predictor_helper(A, B, C)
            rec[i, j] = np.clip(pr + dq[i, j], 0, 255)

    return rec


def DeCompress() :
    # load compressed files
    indexes = np.load("indexes.npy")
    Q_bar = np.load("qbar.npy")
    edges = np.load("edges.npy")
    
    # dequantizer
    dq = Uniform_DeQuantizer(indexes, Q_bar)
    
    # reconstruction
    rec = Reconstruct_Image(dq, edges)

    # show images
    show_images(
                [dq, rec],
                ["DeQuantized", "Reconstructed"]
            )
    
    # save reconstructed image
    # ...................................



def menu() :
    while True :
        print("welcome, enter choice:")
        print("1) Compress.")
        print("2) DeCompress.")
        print("3) Exit.")
        print(">> ", end=" ")
        choice = input()
        if choice == "1" :
            Compress()
        elif choice == "2" :
            DeCompress()
        elif choice == "3" :
            return
        else :
            print("# enter valid choice.")
            menu()



menu()