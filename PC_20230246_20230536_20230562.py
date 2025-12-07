import math
import numpy as np


def Uniform_Quantizer(Array, N_bits = 2) :
    levels = 2**N_bits
    array_min = min(Array)
    array_max = max(Array)
    diff = (array_max - array_min)
    indexs = []
    Q_bar = []
    
    for x in Array :
        # .....................
        pass
        
    return indexs, Q_bar

def Compression_ration() :
    pass


def Compress() :
    # read image 
    
    # predictor
    
    # Error
    
    # Uniform quantizer
    # Uniform_Quantizer(Array ,step_size)
    
    # Compression ration
    Compression_ration()
    


def Uniform_DeQuantizer() :
    pass

def Reconstruct_Image() :
    pass


def DeCompress() :
    # dequantizer
    Uniform_DeQuantizer()
    
    # reconstruction
    Reconstruct_Image()
    pass



def menu() :
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