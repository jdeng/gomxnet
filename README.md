# gomxnet
Amalgamation and go binding

 
## Go binding for predictor
 * Use ```go get github.com/jdeng/gomxnet``` to install. If your openblas version happens to be 0.2.14_1 you are all set.
 * Update ```predict.go```: update ```openblas``` library path accordingly (the first two lines).
```
...
//#cgo CXXFLAGS: -std=c++11 -I/usr/local/Cellar/openblas/0.2.14_1/include
//#cgo LDFLAGS: -L /usr/local/Cellar/openblas/0.2.14_1/lib/ -lopenblas
//#include <stdlib.h>
//#include "mxnet.h"
import "C"
import "unsafe"
...
```
* ```go build``` could be slow when importing this library due to recompiling mxnet.cc every time. One workaround is manually building and linking mxnet.a. You will need to remove mxnet.cc to prevent go build compiling it. See below
```
cd $GOPATH/src/github.com/jdeng/gomxnet/amalgamation
sh build.sh
ls mxnet.a 
rm ../mxnet.cc
```
And add one line to ```predict.go```. It needs to be an absolute path.
```
//#cgo LDFLAGS: /<path-to-gopath>/src/github.com/jdeng/gomxnet/amalgamation/mxnet.a -lstdc++
```
 * Build the sample ```example/main.go``` with ```go build``` in the example directory. You need to install a dependent library with ```go get github.com/disintegration/imaging```.
 ** Download the model file package from [https://github.com/dmlc/mxnet-model-gallery] and update the path in ```main.go```. Build with ```go build```. Try with ```./example cat15.jpg```, the program should be able to recognize the cat.
 * Tested with golang 1.5.1 on Mac OS X Yosemite.
 * Sample usage (from ```src/main.go```)
```
  // read model files into memory
  symbol, _ := ioutil.ReadFile("./Inception-symbol.json")
  params, _ := ioutil.ReadFile("./Inception-0009.params")
  
  // create predictor with model, device and input node config
  batch := 1
  pred, _ := gomxnet.NewPredictor(gomxnet.Model{symbol, params}, gomxnet.Device{gomxnet.CPU, 0}, []gomxnet.InputNode{{"data", []uint32{batch, 3, 224, 224}}})

  // get input vector from 224 * 224 image(s)
  input, _ := gomxnet.InputFrom([]image.Image{img}, gomxnet.ImageMean{117.0, 117.0, 117.0})
  
  // feed forward
  pred.Forward("data", input)
  
  // get the first output node. length for each image is len(output) / batch
  output, _ := pred.GetOutput(0)
  
  // free the predictor
  pred.Free()

```
## mxnet amalgamation (this is optional. A pre-generated mxnet.cc is already in the directory.)
 * Check out mxnet, e.g., in ~/Sources/, update submodules and build
 * Generate ```mxnet.cc``` in ```amalgamation``` directory using ```amalgamation/gen.sh``` (content shown below). You may need to update the first two lines to point to your mxnet and openblas locations.
```
export MXNET_ROOT=~/Source/mxnet
export OPENBLAS_ROOT=/usr/local/Cellar/openblas/0.2.14_1
rm -f ./mxnet
echo "Linking $MXNET_ROOT to ./mxnet"
ln -s $MXNET_ROOT ./mxnet
echo "Generating deps from $MXNET_ROOT to mxnet0.d with mxnet0.cc"
g++ -MD -MF mxnet0.d -std=c++11 -Wall -I ./mxnet/ -I ./mxnet/mshadow/ -I ./mxnet/dmlc-core/include -I ./mxnet/include -I$OPENBLAS_ROOT/include -c  mxnet0.cc

echo "Generating amalgamation to mxnet.cc"
python ./expand.py

cp mxnet.cc ../
echo "Done"
```

# TODO
* Merge openblas into the amalgamation?
* Add Train API?

