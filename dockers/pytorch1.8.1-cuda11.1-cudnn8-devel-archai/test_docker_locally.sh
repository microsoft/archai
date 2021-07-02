# This command maps /home/dedey to /root/dedey inside the docker and drops one into 
# a terminal inside the docker. Then one can just run the training command.
docker run --runtime=nvidia -it --rm -v /home/dedey:/root/dedey debadeepta/pytorch1.8.1-cuda11.1-cudnn8-devel-archai:latest 