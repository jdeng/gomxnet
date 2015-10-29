package gomxnet

//#cgo LDFLAGS: ../../amalgamation/mxnet-all.a -lstdc++ -L /usr/local/Cellar/openblas/0.2.14_1/lib/ -lopenblas
//#include <stdlib.h>
//#include "../../amalgamation/c_predict_api.h"
import "C"
import "unsafe"

import "fmt"
type Predictor struct {
	handle C.PredictorHandle
}

func NewPredictor(symbolFile []byte, paramFile []byte, devType int, devId int, numInputNodes uint32, inputKeys []string, inputShapeInd []uint32, inputShapeData []uint32) (*Predictor, error) {
	var b *C.char
	ptrSize := unsafe.Sizeof(b)
	ik := C.malloc(C.size_t(len(inputKeys)) * C.size_t(ptrSize))
	for i:=0; i<len(inputKeys); i++ {
		element := (**C.char)(unsafe.Pointer(uintptr(ik) + uintptr(i)*ptrSize))
		*element = C.CString(inputKeys[i])
	}

	var handle C.PredictorHandle
	n, err := C.MXPredCreate((*C.char)(unsafe.Pointer(&symbolFile[0])), (*C.char)(unsafe.Pointer(&paramFile[0])), C.size_t(len(paramFile)), C.int(devType), C.int(devId), C.mx_uint(numInputNodes), (**C.char)(ik) , (*C.mx_uint)(&inputShapeInd[0]) , (*C.mx_uint)(&inputShapeData[0]) , &handle)

	for i:=0; i<len(inputKeys); i++ {
		element := (**C.char)(unsafe.Pointer(uintptr(ik) + uintptr(i)*ptrSize))
		C.free(unsafe.Pointer(*element))
	}
	C.free(unsafe.Pointer(ik))
	
	if err != nil { return nil, err }
	if n < 0 {
		return nil, fmt.Errorf("Failed to create predictor") 
	}

	return &Predictor{handle}, nil
}

func (p *Predictor) Free() {
	C.MXPredFree(p.handle)
	p.handle = nil
}

func (p *Predictor) Forward(key string, data []float32) error {
	if data != nil {
		k := C.CString(key)	
		defer C.free(unsafe.Pointer(k))
		if n, err := C.MXPredSetInput(p.handle, k, (*C.mx_float)(&data[0]), C.mx_uint(len(data))); err != nil {
			return err
		} else if n < 0 {
			return fmt.Errorf("Failed to set input") 
		}
	}

	if n, err := C.MXPredForward(p.handle); err != nil {
		return err
	} else if n < 0 {
		return fmt.Errorf("Failed to forward: %d", n)
	}
	return nil
}

func (p *Predictor) GetOutput(index uint32) ([]float32, error) {
	var shapeData *C.mx_uint
	var shapeDim C.mx_uint
	if n, err := C.MXPredGetOutputShape(p.handle, C.mx_uint(index), (**C.mx_uint)(&shapeData), (*C.mx_uint)(&shapeDim)); err != nil {
		return nil, err
	} else if n < 0 {
		return nil, fmt.Errorf("Failed to get output shape: %d", n)
	}

	size := shapeDim //?
	data := make([]C.mx_float, size)
	if n, err := C.MXPredGetOutput(p.handle, C.mx_uint(index), (*C.mx_float)(&data[0]), size); err != nil {
		return nil, err
	} else if n < 0 {
		return nil, fmt.Errorf("Failed to get output: %d", n)
	}
	out := make([]float32, size)
	for i:=0; i<int(size); i++ {
		out[i] = float32(data[i])
	}
	return out, nil
}

func GetLastError() error {
	if err := C.MXGetLastError(); err != nil {
		return fmt.Errorf(C.GoString(err))
	}
	return nil
}
