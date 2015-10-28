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

func NewPredictor(symbolFile string, paramFile string, devType int, devId int, numInputNodes uint32, inputKeys []string, inputShapeInd []uint32, inputShapeData []uint32) (*Predictor, error) {
	sf := C.CString(symbolFile)
	pf := C.CString(paramFile)

	var b *C.char
	ptrSize := unsafe.Sizeof(b)
	ik := C.malloc(C.size_t(len(inputKeys)) * C.size_t(ptrSize))
	for i:=0; i<len(inputKeys); i++ {
		element := (**C.char)(unsafe.Pointer(uintptr(ik) + uintptr(i)*ptrSize))
		*element = C.CString(inputKeys[i])
	}

	var handle C.PredictorHandle
	n, err := C.MXPredCreate(sf, pf, C.int(devType), C.int(devId), C.mx_uint(numInputNodes), (**C.char)(ik) , (*C.mx_uint)(&inputShapeInd[0]) , (*C.mx_uint)(&inputShapeData[0]) , &handle)
	C.free(unsafe.Pointer(sf))
	C.free(unsafe.Pointer(pf))

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
}

func (p *Predictor) Forward() error {
	if n, err := C.MXPredForward(p.handle); err != nil {
		return err
	} else if n < 0 {
		return fmt.Errorf("Failed to forward: %d", n)
	}
	return nil
}

func GetLastError() error {
	if err := C.MXGetLastError(); err != nil {
		return fmt.Errorf(C.GoString(err))
	}
	return nil
}
