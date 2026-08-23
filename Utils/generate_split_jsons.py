import json
import numpy as np
from sklearn.model_selection import KFold
import argparse
with open('split_files/mr_hisum_split.json') as f:
    mr_hisum_splits = json.load(f)

mr_hisum_train_videos =[f'mrhisum/{video}' for video in mr_hisum_splits['train_keys']]
mr_hisum_val_videos =[f'mrhisum/{video}' for video in mr_hisum_splits['val_keys']]
mr_hisum_test_videos =[f'mrhisum/{video}' for video in mr_hisum_splits['test_keys']]

tvsum_videos = [f'tvsum/video_{i}' for i in range(1,51)]
summe_videos = [f'summe/video_{i}' for i in range(1,26)]
# Add VideoXum Here\

dataset_video_list = {'tvsum':tvsum_videos,'summe':summe_videos,'mrhisum':[mr_hisum_train_videos,mr_hisum_val_videos,mr_hisum_test_videos]}
def create_canonical_splits(dataset_name,seed = 15,additional_name=None):
    if dataset_name in ['tvsum','summe']:
        print('Generate Cross Validation Split')
        values = dataset_video_list[dataset_name]
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        all_splits = []
        for split, (train_idx, val_idx) in enumerate(kf.split(values), start=1):
                    train = [values[i] for i in train_idx]
                    validation = [values[i] for i in val_idx]
                    data = {
                            "train": train,
                            "val": validation
                        }
                    all_splits.append(data)
        data = all_splits
    if dataset_name in ['videoxum','mrhisum']: 
        print('Loading Train Test and Validation')
        train,val,test = dataset_video_list[dataset_name][0],dataset_video_list[dataset_name][1],dataset_video_list[dataset_name][2]
        data = [{
                'train':train,
                'val': val,
                'test': test          }]
    
    with open(f"{dataset_name}_can.json", "w") as f:
                                json.dump(data, f, indent=2)


def create_transfer_splits(dataset_name:str,included_datasets:list,seed=15):
        assert len(included_datasets)>1, "Include a dataset"
        train = []
        val = dataset_video_list[dataset_name]
        for included_dataset in included_datasets:
                if included_dataset in ['mrhisum','videoxum']:
                        data_keys = dataset_video_list[included_dataset]
                        # Only include train and validation from other datasets
                        for i in range(len(data_keys)-1):
                                train.extend(data_keys[i])
                else:
                        train.extend(dataset_video_list[included_dataset])
        data = [{
                                    "train": train,
                                    "val": val,
                    
                                }]
        
        with open(f"{dataset_name}_tran_.json", "w") as f:
                json.dump(data, f, indent=2)

def create_augmented_splits(dataset_name:str,included_datasets:list,seed=15):
        assert len(included_datasets)>1, "Include a dataset"

        if dataset_name in ['tvsum','summe']:
                print('Generate Cross Validation Split')
                values = dataset_video_list[dataset_name]
                kf = KFold(n_splits=5, shuffle=True, random_state=seed)
                all_splits = []
                for split, (train_idx, val_idx) in enumerate(kf.split(values), start=1):
                            train = [values[i] for i in train_idx]
                            validation = [values[i] for i in val_idx]
                            for included_dataset in included_datasets:
                                    if included_dataset in ['mrhisum','videoxum']:
                                            data_keys = dataset_video_list[included_dataset]
                                            # Only include train and validation from other datasets
                                            for i in range(len(data_keys)-1):
                                                    train.extend(data_keys[i])
                                    else:
                                            train.extend(dataset_video_list[included_dataset])
                            data = {
                                                        "train": train,
                                                        "val": validation,
                                                    }
                            all_splits.append(data)
                data = all_splits
        if dataset_name in ['videoxum','mrhisum']: 
                print('Loading Train Test and Validation')
                train,val,test = dataset_video_list[dataset_name][0],dataset_video_list[dataset_name][1],dataset_video_list[dataset_name][2]
                for included_dataset in included_datasets:
                                                    if included_dataset in ['mrhisum','videoxum']:
                                                            data_keys = dataset_video_list[included_dataset]
                                                            # Only include train and validation from other datasets
                                                            for i in range(len(data_keys)-1):
                                                                    train.extend(data_keys[i])
                                                    else:
                                                            train.extend(dataset_video_list[included_dataset])
                data = [{
                                'train':train,
                                'val': val,
                                'test': test
                          }]
        with open(f"{dataset_name}_aug.json", "w") as f:
                                    json.dump(data, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Split Generation ")
    parser.add_argument(
        "--split_type",
        type=str,
        required=True,
        help="The type of split to include (canonical), (augmented), (transfer)"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The dataset to include"
    )
    parser.add_argument(
            "--included_datasets",
            type=list,
            default=None,
            help="Additional dataset to include for augmented and transfer"
        )
    parser.add_argument(
                "--seed",
                type=int,
                default = 15,
                help="The seed to run"
            )
    return parser.parse_args()

def main():
    args = parse_args()
    if args['split_type'] == 'can':
            dataset_name = args.dataset_name
            seed = args.seed
            create_canonical_splits(dataset_name,seed)
    elif args['split_type'] =='aug':
            dataset_name = args.dataset_name
            seed = args.seed
            included_datasets = args.included_datasets
            create_augmented_splits(dataset_name,included_datasets,seed)
    elif args['split_type'] =='trans':
                dataset_name = args.dataset_name
                seed = args.seed
                included_datasets = args.included_datasets
                create_transfer_splits(dataset_name,included_datasets,seed)

if __name__ == "__main__":
        main()