docker run --gpus all --name archai \
    --rm \
    -u $(id -u):$(id -g) \
    -e HOME=$HOME -e USER=$USER \
    -v $HOME:$HOME \
    -v /dataroot:$HOME/dataroot \
    -w $HOME \
    --ipc=host \
    --net=host \
    -it sytelus/archai