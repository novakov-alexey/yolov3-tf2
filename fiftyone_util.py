import fiftyone as fo
import click


def load_dataset(name: str):
    # dataset_dir = "/Users/Alexey_Novakov/fiftyone/open-images-v6-validation-100"
    return fo.load_dataset(name)
    # return fo.Dataset.from_dir(
    #     dataset_dir, fo.types.FiftyOneImageDetectionDataset, name="open-images-v6-train-100"
    # )


def export_tf_record(dataset: fo.core.dataset.Dataset, export_dir: str):
    label_field = "ground_truth"  # for example

    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.TFObjectDetectionDataset,
        # label_field=label_field,
    )


def launch_app(dataset: fo.core.dataset.Dataset):
    session = fo.launch_app(dataset, port=5151)
    session.wait()


# "validation"
def load_zoo_dataset(split: str, max_samples: int):
    dataset = fo.zoo.load_zoo_dataset(
        "open-images-v6",
        split=split,
        label_types=["detections"],
        classes=["Toy"],
        max_samples=max_samples,
    )
    dataset.persistent = True
    return dataset


@click.command("convert")
def convert():
    dataset = load_dataset("open-images-v6-validation-200")
    export_dir = "data/open-images-v6-train-200"
    export_tf_record(dataset, export_dir)

    dataset = load_dataset("open-images-v6-train-800")
    export_dir = "data/open-images-v6-train-800"
    export_tf_record(dataset, export_dir)


@click.command("launch")
def launch():
    dataset = load_dataset("open-images-v6-train-800")
    launch_app(dataset)


@click.command("load_datasets")
def load_datasets():
    dataset = load_zoo_dataset("train", 800)
    dataset = load_zoo_dataset("validation", 200)


@click.group()
def cli():
    pass


cli.add_command(convert)
cli.add_command(launch)
cli.add_command(load_datasets)

if __name__ == "__main__":
    cli()
