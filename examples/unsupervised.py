from betterloader import UnsupervisedBetterLoader
from betterloader.defaults import collate_metadata
from PIL import Image

index_json = "./sample_index_unsupervised.json"
basepath = "./sample_dataset/"
batch_size = 2
metadata = collate_metadata()
better_loader = UnsupervisedBetterLoader(
    basepath=basepath,
    base_experiment_details=["simclr", 1, (150, 150)],
    index_json_path=index_json,
    dataset_metadata=metadata,
)
dataloaders, sizes = better_loader.fetch_segmented_dataloaders(batch_size=batch_size)

for i, ((xp1, xp2), _) in enumerate(dataloaders["train"]):
    print(i)
    print(xp1.shape)
