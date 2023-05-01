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

