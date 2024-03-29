import tensorflow as tf

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=5000,
        seed=1234,
        fp16_run=False,
        multi_gpu=False,
        max_checkpoints_to_keep=5,
        cudnn_enabled=True,
        cudnn_benchmark=False,
        batch_size=2,
        gpu_id=0,

        ################################
        # Data Parameters             #
        ################################
        training_files='filelists/obama_text_train_filelist.txt',
        validation_files='filelists/obama_text_val_filelist.txt',
        text_cleaners=['english_cleaners'],
        checkpoint_path="checkpoints",
        checkpoint=None,
        csv_dir="/hdd/open-images/csvs",
        train_csv_file="data/csvs/train-annotations-bbox-collapsed.csv",
        train_images_dir = "data/unzipped/all_train",
        # dataset_size = None,
        # dataset_size=2000,
        dataset_size=50000,

        ################################
        # Model Parameters             #
        ################################
        max_objects_per_image=100,
        min_images_per_class=1,
        image_size=128
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
