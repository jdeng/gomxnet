export MXNET_ROOT=~/Source/mxnet
rm -f ./mxnet
ln -s $MXNET_ROOT ./mxnet
echo "Generating deps from $MXNET_ROOT to mxnet.d with mxnet.cc"
g++ -MD -MF mxnet.d -std=c++11 -Wall -I ./mxnet/ -I ./mxnet/mshadow/ -I ./mxnet/dmlc-core/include -I ./mxnet/include -I/usr/local/Cellar/openblas/0.2.14_1/include -c  mxnet.cc

echo "Generating amalgamation to mxnet-all.hh and mxnet-all.cc"
python ./expand.py

rm ./mxnet
echo "Done"


