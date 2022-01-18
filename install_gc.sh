gitrootdir=`git rev-parse --show-toplevel`

pushd $gitrootdir/src/experiments/delphi/compiled_gcs/
cat gcs* > gcs.tar.gz
tar -xzf gcs.tar.gz
rm -rf gcs.tar.gz
mv gcs/* ../
rm -rf gcs

# Lenet gcs
#pushd $gitrootdir/src/experiments/delphi/compiled_gcs/lenet
#cat lenet* > lenet.tar.gz
#tar -xzf lenet.tar.gz
#rm -rf lenet.tar.gz
#rm -rf ../../lenet_mnist
#mv lenet_mnist/ ../../
#rm -rf lenet_mnist/
#
## Cifar10
#pushd $gitrootdir/src/experiments/delphi/compiled_gcs/resnet_32/
#cat resnet_32* > resnet_32.tar.gz
#tar -xzf resnet_32.tar.gz
#rm -rf resnet_32.tar.gz
#rm -rf ../../resnet_32_cifar10
#mv resnet_32_cifar10 ../../
#rm -rf resnet_32_cifar10
#
## Cifar100 gcs
#pushd $gitrootdir/src/experiments/delphi/compiled_gcs/resnet_34/
#cat resnet_34* > resnet_34.tar.gz
#tar -xzf resnet_34.tar.gz
#rm -rf ../../resnet_34_cifar100
#rm -rf resnet_34.tar.gz
#mv resnet_34_cifar100 ../../
#rm -rf resnet_34_cifar100
#
## vgg Cifar100 gcs
#pushd $gitrootdir/src/experiments/delphi/compiled_gcs/vgg_cifar100/
#cat vgg_cifar100* > vgg_cifar100.tar.gz
#tar -xzf vgg_cifar100.tar.gz
#rm -rf ../../vgg_cifar100
#rm -rf vgg_cifar100.tar.gz
#mv vgg_cifar100 ../../
#rm -rf vgg_cifar100

