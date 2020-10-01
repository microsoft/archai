del /q source\archai*.rst
sphinx-apidoc -f -o ../docs/ ../archai/ ../archai/cifar10_models/** ../archai/data_aug/** ../archai/netowrks/**
make html