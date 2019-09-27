import albumentations as A

def train_albu_multi_augment(image, mask):
    return image, mask 

###########################################################################################

def albu_augment_default(image, mask=None):
    return image, mask

def albu_augment_normalize(image, mask=None):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    augment = A.Normalize(mean, std)
    image = augment(image=image)['image']
    
    return image, mask
