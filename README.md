# gomxnet
Amalgamation and go binding

## mxnet amalgamation
 * Check out mxnet, e.g., in ~/Sources/, update submodules and build
 * Generate mxnet-all.cc in ```amalgamation``` directory using ```amalgamation/gen.sh``` (content shown below). You may need to update the first line to point to your mxnet directory.
```
export MXNET_ROOT=~/Source/mxnet
rm -f ./mxnet
echo "Linking $MXNET_ROOT to ./mxnet"
ln -s $MXNET_ROOT ./mxnet
echo "Generating deps from $MXNET_ROOT to mxnet.d with mxnet.cc"
g++ -MD -MF mxnet.d -std=c++11 -Wall -I ./mxnet/ -I ./mxnet/mshadow/ -I ./mxnet/dmlc-core/include -I ./mxnet/include -I/usr/local/Cellar/openblas/0.2.14_1/include -c  mxnet.cc

echo "Generating amalgamation to mxnet-all.cc. Use build.sh to generate mxnet-all.a"
python ./expand.py
echo "Done"
```
 * Build static libary ```mxnet-all.a``` with ```amalgamation/build.sh``` (content shown below). You'll need to install openblas (e.g., in Mac OS X with ```brew install openblas```) and update the include path below.
```
echo "Building mxnet-all.a from mxnet-all.cc"
export CXX=g++
$CXX -O0 -g -I/usr/local/Cellar/openblas/0.2.14_1/include -std=c++11 -Wall -o mxnet-all.o -c mxnet-all.cc 
ar rcs mxnet-all.a mxnet-all.o
```

## Go binding for predictor
 * Update ```src/gomxnet/predict.go```. Point to your static library ```mxnet-all.a``` and update ```openblas``` library path accordingly.
```
...
//#cgo LDFLAGS: /Users/jack/Work/gomxnet/amalgamation/mxnet-all.a -lstdc++ -L /usr/local/Cellar/openblas/0.2.14_1/lib/ -lopenblas
//#include <stdlib.h>
//#include "../../amalgamation/c_predict_api.h"
import "C"
...
```
 * Build the sample ```main.go``` with ```go build```. You need to install a dependent library with ```go get github.com/disintegration/imaging```
 * Tested with golang 1.5
 * Sample usage
```
  // read model files into memory
  symbol, _ := ioutil.ReadFile("../Inception-symbol.json")
  params, _ := ioutil.ReadFile("../Inception-0009.params")
  
  // create predictor with model, device and input node config
  pred, _ := gomxnet.NewPredictor(gomxnet.Model{symbol, params}, gomxnet.Device{gomxnet.CPU, 0}, []gomxnet.InputNode{{"data", []uint32{1, 3, 224, 224}}})

  // get input vector from image (should be 224x224)
  input, _ := gomxnet.InputFrom(img, 117.0, 117.0, 117.0)
  
  // feed forward
  pred.Forward("data", input)
  
  // get the first output node
  output, _ := pred.GetOutput(0)
  
  // free the predictor
  pred.Free()

```
