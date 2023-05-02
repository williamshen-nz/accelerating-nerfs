# accelerating-nerfs
NeRFs are slow, we are trying to accelerate them!

## General Setup
To train and render NeRFs. We use nerfacc to speed things up to make our life easier.
```
conda create -n accnerfs python=3.8
conda activate accnerfs
pip install -r requirements.txt
```

## Timeloop and Accelergy Setup
We use Timeloop and Accelergy to evaluate our designs.
```
export DOCKER_ARCH=amd64

# If you are using arm CPU (Apple M1/M2), 
# export DOCKER_ARCH=arm64 

docker-compose up
```

### Download NeRF datasets and checkpoints
**Datasets**
1. Download `nerf_synthetic.zip` from https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
2. Unzip `nerf_synthetic.zip` and update the `data_root` variable in the scripts.

**Checkpoints**
1. Download checkpoints from https://drive.google.com/file/d/1vw9H-5xXYr6Q_tHcpVc0Kri96i6Do4vE/view?usp=share_link
2. Unzip to the project directory (at the level of README.md)

**Expected Directory Structure**
```
accelerating-nerfs (project root)
├── accelerating_nerfs/
├── README.md
├── nerf_synthetic/
├── nerf-synthetic-checkpoints/
```
