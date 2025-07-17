import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode as Interpolation

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
CONCH_MEAN = (0.48145466, 0.4578275, 0.40821073)
CONCH_STD = (0.26862954, 0.26130258, 0.27577711)
UNI_MEAN = (0.485, 0.456, 0.406) 
UNI_STD = (0.229, 0.224, 0.225)

basic_img = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ]
)
aug_img = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ]
)
basic_open = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    ]
)
aug_open = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    ]
)
basic_M = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ToTensorV2(),
    ]
)

aug_M = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.ColorJitter(brightness=0.2, contrast=0.8, saturation=0.8, hue=0.2, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=2, p=0.5),
        A.GaussianBlur(p=0.4),
        A.ImageCompression(p=0.1),
        A.ToGray(p=0.3),
        A.Sharpen((0.2, 0.5), (0.5, 1.0), p=0.2),
        A.ChannelShuffle(p=0.4),
        A.Downscale(scale_min=0.5, scale_max=0.5, p=0.2),
        A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ToTensorV2(),
    ]
)

basic_conch = T.Compose(
    [
        T.Resize(size=224, interpolation=Interpolation.BICUBIC, max_size=None, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=CONCH_MEAN, std=CONCH_STD),
    ]
)

aug_conch = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=CONCH_MEAN, std=CONCH_STD),
    ]
)

basic_uni = T.Compose(
    [
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=UNI_MEAN, std=UNI_STD),
    ]
)

aug_uni = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=UNI_MEAN, std=UNI_STD),
    ]
)

basic_biomed = T.Compose(
    [
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=CONCH_MEAN, std=CONCH_STD),
    ]
)

aug_biomed = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=CONCH_MEAN, std=CONCH_STD),
    ]
)

basic_virchow2 = T.Compose(
    [
        T.Resize(size=224, interpolation=Interpolation.BICUBIC, max_size=None, antialias=True),
        T.CenterCrop(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean = UNI_MEAN, std = UNI_STD)
    ]
)

aug_virchow2 = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean = UNI_MEAN, std = UNI_STD)
    ]
)


basic_gpfm = T.Compose(
    [
        T.Resize(size=224, interpolation=Interpolation.BICUBIC, max_size=None, antialias=True),
        T.ToTensor(),
        T.Normalize(mean = UNI_MEAN, std = UNI_STD)
    ]
)

aug_gpfm = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean = UNI_MEAN, std = UNI_STD)
    ]
)

basic_mstar = T.Compose(
    [
        T.Resize(size=224),
        T.ToTensor(),
        T.Normalize(mean = UNI_MEAN, std = UNI_STD)
    ]
)

aug_mstar = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean = UNI_MEAN, std = UNI_STD)
    ]
)