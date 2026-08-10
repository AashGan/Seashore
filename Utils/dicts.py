import h5py

dataset_metadata_paths = {'tvsum','summe','videoxum','mrhisum'}

class MetadataStore:
    def __init__(self, datasets,dataset_paths_dict:dict = None):
        # The dataset dict paths should also allow you to override and add custom h5's incase the h5's deviate (different shot boundaries,fps sampling etc)

        self.datasets = datasets
        self.dataset_path_dicts = dataset_paths_dict
        self.files = {}

    def open(self):
        if self.dataset_path_dicts:
            self.files = {dataset:h5py.File(paths) for dataset,paths in self.dataset_path_dicts.items()}
        else:
            self.files = {
                dataset: h5py.File(
                    f"Data/Metadata/{dataset}_metadata.h5", "r"
                )
                for dataset in self.datasets
            }

    def get(self, dataset, video_key):
        f = self.files[dataset]
        group = f[video_key]

        metadata = {
            "positions": group["positions"][...],
            "n_frames": int(group["n_frames"][...]),
            "shot_bounds": group["shot_bounds"][...]
        }


        return metadata

    def close(self):
        for f in self.files.values():
            f.close()
        self.files.clear()