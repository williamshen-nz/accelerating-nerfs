# Note: architectures moved to notebooks under top-level directory

Final Project Baselines
------------------------------------------
This repo contains some baseline implementations of an Eyeriss-like architecture,a simba-like (NVDLA style) architecture,
a simple weight stationary architecture, a processing-in-memory architecture, and an output stationary architecture.
Please find them in the `designs` folder. *Note that the processing-in-memory architecture folder contains [a link to another repo](https://github.com/nelliewu95/processing-in-memory-example).*

### System requirement

We use the same docker as the previous labs.
Please perform a `pull` of the docker image before you start the project. 
And run the following command inside the directory that contains the docker-compose.yaml file to get the latest docker:
  
```
docker-compose pull
```


If you run `docker-compose.yaml` for labs using `docker-compose up`, Jupyter Notebook will be launched automatically. If you want to use Timeloop-Accelergy commands from terminal directly, use `docker run` instead. For example:
```
docker run --rm -it mitdlh/timeloop-accelergy-pytorch:latest bash
```
You will see logs for Jupyter Notebook in terminal, but you can `ctrl+C` to stop Jupyter server and start using commands directly. 

### File Structure
- example_designs: 
   - architecture descriptions, compound component descriptions, 
  constraints descriptions, mapper descriptions for the example designs.
   - note that for each architecture, we have 2 constraint files:
        1. *_arch_constraints.yaml describes the necessary hardware and dataflow constraints
        2. *_map_constraits.yaml describes the map space optimizations for the  designs
- layer_shapes: 
    - Example workloads: AlexNet, VGG01, VGG02
- scripts
    - A set of scripts for generating your own workloads in Timeloop format
    - Instructions:
        - `cd scripts`
        - modify the `cnn_layers.py` file to describe your own workload
        - `python3 construct_workloads.py <my_dnn_model_name>`
    - Please refer to the repository for [PyTorch2Timeloop Converter](https://github.com/Accelergy-Project/pytorch2timeloop-converter) for automating the workload conversion.

### Run simulations

To run a simulation using timeloop-accelergy system you should `cd` to the specific design's folder and run timeloop
simulations on a specific workload

Here is an example for running AlexNet Layer1 on the `simple_weight_stationary` architecture: 
```
cd example_designs/simple_weight_stationary
timeloop-mapper arch/simple_weight_stationary.yaml arch/components/*.yaml mapper/mapper.yaml constraints/*.yaml ../../layer_shapes/AlexNet/AlexNet_layer1.yaml
```

You will see the following outputs generated:
- timeloop-mapper.accelergy.log: accelergy's runtime info while generating the ERT.
- timeloop-mapper.ART.yaml: the area reference table generated by Accelergy for the architecture
- timeloop-mapper.ART_summary.yaml: the area reference table for the components as well as the associated plug-ins used for generating the outputs.
- timeloop-mapper.ERT.yaml: the energy reference table generated by Accelergy for the architecture.
- timeloop-mapper.ERT_summary.yaml: the energy reference table for the components as well as the associated plug-ins used for generating the outputs.
- timeloop-mapper.flattened_architecture.yaml: the fully defined and flattened architecture interpreted by Accelergy.
- timeloop-mapper.log: timeloop runtime info.
- timeloop-mapper.map.txt: the best mapping found by Timeloop mapper.
- timeloop-mapper.stats.txt: the runtime behaviors of the components in the design generated by Timeloop.
- timeloop-mapper.ma+stats.xml: the raw information generated by Timeloop -- you don't need to read into it.

If you prefer to run the simulation in background without the interactive thread updates, please trun of the `live-status` flag in
`mapper/mapper.yaml` in each design folder.

** Note that for the provided designs and workloads, your simulation should generally converge within 30 mins. Once you see
the simulations converging, you can press `ctrl + C` to manually stop them. They sometimes will take much longer to 
automaticaly stop as we set the converging cretiria to be pretty high to avoid early-stop with subooptimal mappings. 
Please use you own judgement. **


### How to use these baselines?
As described in the final project information slides, you can use the different baselines for different final project ideas.
 
*If your final project needs simulation/design support that is not currently available in the provided framework/examples,
 please reach out to the course staff **BEFORE** you get started on designing the project*.

###  Related reading
 - [Timeloop/Accelergy documentation](https://timeloop.csail.mit.edu/)
 - [Timeloop/Accelergy tutorial](http://accelergy.mit.edu/tutorial.html)
 - [SparseLoop tutorial](https://accelergy.mit.edu/sparse_tutorial.html)
 - [eyeriss-like design](https://people.csail.mit.edu/emer/papers/2017.01.jssc.eyeriss_design.pdf)
 - [simba-like architecture](https://people.eecs.berkeley.edu/~ysshao/assets/papers/shao2019-micro.pdf)
 - simple weight stationary architecture: you can refer to the related lecture notes
 - simple output stationary architecture: you can refer to the related lecture notes
