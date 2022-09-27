import os
import time
import warnings

from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ['CUDA_LAUNCH_BLOCKING'] = "3"
CUDA_LAUNCH_BLOCKING = 1
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

from common.ClassFile import ClassFile


class GbcCnnService:
    DEVICE_TYPE = 'cuda:0'
    DATA_DIR = "./../classification/dataset"
    MODEL_PATH = './model'
    MODEL_LOAD_NAME = '_img_v2.model'
    MODEL_SAVE_NAME = '_img_v3.model'
    NUM_EPOCHS = 15
    BATCH_SIZE = 128
    SPLIT = .2

    def __init__(self):
        self.data_train_loader = None
        self.data_validation_loader = None
        self.path_to_load_model = os.path.join(self.MODEL_PATH, self.MODEL_LOAD_NAME)
        self.path_to_save_model = os.path.join(self.MODEL_PATH, rf"{self.MODEL_SAVE_NAME}.all")
        self.inputs = None
        self.class_list = list()
        self.class_name_list = list()
        self.dataset_size_dict = dict()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.device = torch.device(self.DEVICE_TYPE)

    def make_model(self):
        if not ClassFile.has_file(self.MODEL_PATH, self.MODEL_LOAD_NAME):
            print(f"Loading base model from pre-trained 'VGG11'...")
            self.model = models.vgg11_bn(pretrained=True)

            # fixed pre-trained cnn weights
            for param in self.model.parameters():
                param.requires_grad = False

            # add new classification layer
            self.model.classifier[-1] = nn.Linear(
                in_features=4096, out_features=len(self.class_name_list))

            self.model = self.model.to(self.device)
            torch.save(self.model, self.path_to_save_model)
            print(f"Saving model from '{self.path_to_save_model}'...")
        else:
            print(f"Loading trained model from '{self.path_to_load_model}'...")
            self.model = torch.load(self.path_to_load_model, map_location=self.device)
            self.model.eval()
            self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.model.classifier.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        current_device = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"Current device {current_device} is {torch.cuda.is_available()}")

    def load_data(self):
        """
        read dataset
        """
        print(f"\nPre-process data from '{GbcCnnService.DATA_DIR}'")
        data_transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        print(f"Loading data...")
        image_datasets = datasets.ImageFolder(self.DATA_DIR, data_transform)

        dataset_size = len(image_datasets)
        split_size = int(dataset_size * self.SPLIT)
        train_dataset, val_dataset = \
            random_split(image_datasets, [dataset_size - split_size, split_size])

        self.load_train_data(train_dataset)
        self.load_validation_data(val_dataset)

    def load_train_data(self, train_dataset):
        self.data_train_loader = {x: DataLoader(
            train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=2)
            for x in ['train']}

        self.dataset_size_dict = {x: len(train_dataset) for x in ['train']}
        print(f"Total of training data: {self.dataset_size_dict['train']}")
        self.class_name_list = train_dataset.dataset.classes
        label_dict = {}
        for index, possible_label in enumerate(self.class_name_list):
            label_dict[possible_label] = index
        print(f"\nClasses: {label_dict}")

        # Get a batch of training data to show
        self.inputs, self.class_list = next(iter(self.data_train_loader['train']))

    def load_validation_data(self, val_dataset):
        self.data_validation_loader = {x: DataLoader(
            val_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=2)
            for x in ['val']}
        self.data_train_loader.update(self.data_validation_loader)

        self.dataset_size_dict.update({x: len(val_dataset) for x in ['val']})
        print(f"Total of validation data: {self.dataset_size_dict['val']}")
        self.class_name_list = val_dataset.dataset.classes
        # print(f"Classes: {self.class_name_list}")

        # Get a batch of training data to show
        self.inputs, self.class_list = next(iter(self.data_validation_loader['val']))

    def train_model(self, _num_epochs):
        print(f"Training model...")
        since = time.time()
        best_acc = 0.0

        for epoch in range(1, _num_epochs+1):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                loop = tqdm(enumerate(self.data_train_loader[phase]),
                            leave=False, total=len(self.data_train_loader[phase]))
                loop.set_description(f"Epoch [{epoch}/{_num_epochs}]")
                for batch_idx, (_inputs, labels) in loop:
                    _inputs = _inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(_inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * _inputs.size(0)
                    running_corrects += torch.sum((preds == labels.data))

                    loop.set_description(
                        f"Epoch [{epoch}/{_num_epochs}] - Batch [{batch_idx}]")
                    loop.set_postfix(
                        loss=float(running_loss / self.dataset_size_dict[phase]),
                        acc=float(running_corrects / self.dataset_size_dict[phase]))

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_size_dict[phase]
                epoch_acc = running_corrects / self.dataset_size_dict[phase]

                if phase == "train":
                    print('{} Loss: {:.4f} Acc: {:.4f}'
                          .format(phase.upper(), epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'train' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    score = running_corrects / self.dataset_size_dict[phase]
                    print(f"Best model so far: {score}")
                    torch.save(self.model, self.path_to_save_model)
                    print(f"Saving trained model from '{self.path_to_save_model}'...")

            score_train = running_corrects / self.dataset_size_dict["train"]
            score_val = running_corrects / self.dataset_size_dict["val"]
            if (score_train + score_val) // 2 >= 0.98:
                print("Run ends with 98% accuracy")
                break

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))


def main():
    image_service = GbcCnnService()

    image_service.load_data()

    image_service.make_model()

    image_service.train_model(_num_epochs=image_service.NUM_EPOCHS)


if __name__ == '__main__':
    main()
