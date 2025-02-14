# How to train AerialFormer in Lambda Labs A6000 (1x) instance:

AerialFormer requires an older version of MMCV and MMSegmentation, which in turn
means we need an older version of Pytorch than what is installed on the Lambda
instances. We install and run everything in Docker to take care of these
dependency issues and hopefully to make deployment more portable. These 
instructions assume that there is an appropriate Docker image in AWS ECR
`aerialformer-image` repository based on the Dockerfile in this project

## Preparing the instance for Docker operations

User accounts in Lambda Lab are not part of the docker group when starting a new 
instance, so docker commands will fail with a permissions error. To resolve this
open a terminal and execute:

`sudo adduser "$(id -un)" docker`

Then exit the terminal and open a new one before executing docker commands

## Pull the prepared image from the AWS ECR repository

Note: these instructions could be improved and made more secure by installing
the AWS command line tool on the remote LL instance. There might be a challenge
with having to have access to the browser to perform `aws sso login` so we're 
deferring that for now.

Instead, to get the ECR password, refresh AWS credential in the local workstation
and execute this in a terminal:

`aws ecr get-login-password --region us-west-2`

Copy the password to the clipboard. In the remote terminal login to the ECR repo:

`docker login --username AWS --password PASTED_PASSWORD 654654581134.dkr.ecr.us-west-2.amazonaws.com`

Then pull the latest image using the most recent tag in place of "v1.0.1" as needed:

`docker pull 654654581134.dkr.ecr.us-west-2.amazonaws.com/aerialformer-image:v1.0.1`

## Preparing the container

Run the image in a container named aerialformer. If there is already a container by this
name you might want to `docker rm CONTAINER_ID` to generate a fresh one.

`docker run --ipc=host --gpus=all -d --name=aerialformer 654654581134.dkr.ecr.us-west-2.amazonaws.com/aerialformer-image:v1.0.1 /bin/bash`

Explanation:

**--ipc=host** resolves an error where the container doesn't have enough shared memory to run
the training job (see https://github.com/ultralytics/yolov3/issues/283)

**--gpus=all** ensures that the container has access to the NVIDIA GPUs

**-d** detaches the running container from the terminal for now

## Copy raw data from the solarplane-fs0 to the docker container

`docker cp solarplane-fs0/raw_data/. aerialformer:AerialFormer/raw_data`

The commands that follow are to be executed within the container environment
To get terminal access to the container environment execute:

`docker exec -it aerialformer /bin/sh`

Now you're inside the container. To exit back to the terminal of the
host instance use CTRL-D.

## Edit the potsdam config to change the data_root to 'data/potsdam'
Note: this should probably be fixed in the published image. For now
edit line 3 in the following file:

`vim configs/_base_/datasets/5_potsdam.py`

It should read:

`data_root = 'data/potsdam'` (delete the 5)

## Convert the raw data and place it in the /data dir (takes a couple mins)

`python tools/convert_datasets/potsdam_no_clutter.py raw_data`

## Kick off a training job

`python tools/train.py configs/aerialformer/aerialformer_tiny_512x512_5_potsdam.py`

This will run 160000 iterations in epochs of 5000. After each epoch the model
is checked against the val data set to see how the training is proceeding and a 
report like the following will display in the terminal:

``` 
+--------------------+-------+-------+
|       Class        |  IoU  |  Acc  |
+--------------------+-------+-------+
| impervious_surface |  89.5 | 94.74 |
|      building      | 94.55 | 96.67 |
|   low_vegetation   | 79.18 | 87.98 |
|        tree        | 79.85 | 89.77 |
|        car         | 89.16 | 91.16 |
|      clutter       |  nan  |  nan  |
+--------------------+-------+-------+
2025-02-12 17:23:03,334 - mmseg - INFO - Summary:
2025-02-12 17:23:03,334 - mmseg - INFO - 
+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 92.83 | 86.45 | 92.07 |
+-------+-------+-------+
```
If the mIoU has improved after an epoch MMSegmentation will also save the 
current weights as the new best checkpoint.

## Move checkpoint files to permanent storage

The results of the training are checkpoint files stored in `work_dirs`. 
These need to be moved to the persistent file system before shutting down
the Lambda Labs instance or the files will be lost. 

While the terminal is still inside the Docker container, check the path of
the new checkpoints directory just created in `AerialFormer/work_dirs`.

Then to copy the files, CTRL-D out of the container environment and execute:
`docker cp CONTAINER_ID:AerialFormer/work_dirs/PATH_TO_CHECKPOINTS ~/solarplane-fs0/checkpoints`

## Exporting files to S3

To get files out of Lambda Labs entirely, first make sure they're in the
persistent file system. Then use a tools like [s5cmd](https://github.com/peak/s5cmd)
to export them. 

### Installing `s5cmd` 

From a working SSH terminal session, download the latest AMD64.deb release from
the [s5cmd releases page](https://github.com/peak/s5cmd/releases). E.g.

`wget https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_linux_amd64.deb`

Then:

`sudo apt install ./s5cmd_2.3.0_linux_amd64.deb`

Alternatively, there is an install_s5cmd.sh script in the persistent `/solarplane-fs0`
file system that installs version 2.3.0 directly.

### AWS Access

1. Log into the AWS Console and get current credentials. Copy the commands for
setting environment variables, starting with "export AWS_...".
2. `vim ~/.bashrc` and paste the credentials at the bottom of the file, replacing
and expired credentials found there. Save and exit vim.
3. `source ~/.bashrc` to reinitialize the terminal with the login credentials
4. Navigate to `solarplane-fs0/checkpoints` and ls to get the name of the 
checkpoints directory that you want to export to S3. E.g. "2025_0212_1628"
5. Run the following s5cmd command:

`s5cmd cp 'CHECKPOINTS_DIR/*' 's3://solarplane-model-checkpoints/AerialFormer/CHECKPOINTS_DIR/'`