del /q ..\docs\api\archai*.rst
sphinx-apidoc -H "APIs" -f -o ../docs/api ../archai/ ../archai/cifar10_models/** ../archai/data_aug/** ../archai/netowrks/**
make html