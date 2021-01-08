""" Module for the data loading pipeline for the model to train """


def get_data_loader(dataset, batch_size=None, num_workers=0, seed=42, pin_memory=True, shuffle=True, drop_last=True,
                    batch_sampler=None):
    """
    generate the data_loader from the given dataset
    :param dataset: dataset for training (Should be a PyTorch dataset)
                    Make sure every item is an Image
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => train_loader for the dataset
    """
    from torch.utils.data import DataLoader

    worker_init = None  # lambda worker_id: np.random.seed(seed + worker_id)  # each worker has always the same seed

    if batch_sampler is not None:
        batch_size = 1
        shuffle = False
        drop_last = False

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
        drop_last=drop_last,
        batch_sampler=batch_sampler
    )

    return dl
