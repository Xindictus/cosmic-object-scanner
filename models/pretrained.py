import os
import torch
import torch.optim as optim
import torch.utils as utils
import torchvision

from coco_dataset import CocoDataset
from PIL import Image
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchsummary import summary
from visualization import (
    visualize_prediction,
    visualize_prediction_v2
)


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ANNOTATIONS_PATH = f'{CURRENT_DIR}/../data/coco/result.json'
IMG_PATH = f'{CURRENT_DIR}/../data/coco/'
EPOCHS = 1
BATCH_SIZE = 16
MODEL_PATH = f'{CURRENT_DIR}/fasterrcnn_resnet50_fpn_{EPOCHS}_{BATCH_SIZE}.pth'


def collate_fn(batch):
    return tuple(zip(*batch))


# Function to get the model
def get_model(num_classes):
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        # pretrained=True
    )
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_category_names():
    return {
        0: "Galaxy",
        1: "Nebula",
        2: "Star Cluster"
    }


def model_train(model, all_transforms, device):
    # dataset
    dataset = CocoDataset(IMG_PATH, ANNOTATIONS_PATH, all_transforms)
    data_loader = utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # parameters
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
    # optimizer = optim.RMSprop(
        params,
        lr=0.009,
        # momentum=0.9,
        weight_decay=0.0005
    )

    # print the model summary
    summary(model, input_size=(3, 608, 608))

    # training loop
    for epoch in range(EPOCHS):
        model.train()

        i = 0

        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % BATCH_SIZE == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {losses.item()}")
            i += 1

    print("Training completed")

    # save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved!")

    return model


def main():
    # 3 classes (galaxy, nebula, star-cluster) + background
    num_classes = 4

    model = get_model(num_classes)

    # define the transformation
    all_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if the trained weights file exists
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(device)
        print("Model weights loaded. Skipping training.")
        # Check the device of the model's parameters
        # device = next(model.parameters()).device
        # print(f"Model is on device: {device}")
    else:
        model = model_train(model, all_transforms, device)

    model.eval()
    # Load the image
    # galaxies
    # img_path = f'{IMG_PATH}/images/00c1a7d2-1378.jpg'
    # img_path = f'{IMG_PATH}/images/4b2821f0-2205.jpg'
    # img_path = f'{IMG_PATH}/images/f41c7af9-2054.jpg'
    # img_path = f'{IMG_PATH}/images/5b381b61-2885.jpg'
    # star-cluster
    # img_path = f'{IMG_PATH}/images/efd97f0f-986.jpg'
    # img_path = f'{IMG_PATH}/images/fbf61c19-988.jpg'
    # nebula
    # img_path = f'{IMG_PATH}/images/f3b6d9c5-3616.jpg'
    # img_path = f'{IMG_PATH}/images/fe8ed0ef-957.jpg'
    img_path = f'{IMG_PATH}/images/f461d039-4270.jpg'
    img = Image.open(img_path).convert("RGB")

    # List all files in the directory
    # files = os.listdir(f'{IMG_PATH}/images')

    # # Initialize an empty set to store unique values
    # unique_values = set()

    # for file in files:
    #     img_path = f'{IMG_PATH}/images/{file}'
    #     img = Image.open(img_path).convert("RGB")
    #     # Define the transformation
    #     img_tensor = all_transforms(img).unsqueeze(0).to(device)

    #     # Make the prediction
    #     with torch.no_grad():
    #         prediction = model(img_tensor)

    #         unique_values.update(prediction[0]['labels'])

    #         # if 0 in prediction[0]['labels']:
    #         #     print(file)
    #         #     break

    # print(unique_values)
    # Define the transformation
    img_tensor = all_transforms(img).unsqueeze(0).to(device)

    # Make the prediction
    with torch.no_grad():
        prediction = model(img_tensor)

    # Visualize the prediction
    # visualize_prediction(img, prediction)
    visualize_prediction_v2(
        image=img,
        prediction=prediction,
        threshold=0.6,
        category_names=get_category_names()
    )


if __name__ == '__main__':
    main()
