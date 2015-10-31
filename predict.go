package predict 

//#cgo LDFLAGS: amalgamation/mxnet-all.a -lstdc++ -L /usr/local/Cellar/openblas/0.2.14_1/lib/ -lopenblas
//#include <stdlib.h>
//#include "amalgamation/c_predict_api.h"
import "C"
import "unsafe"

import "fmt"

const (
	CPU = iota + 1
	GPU
	CPU_PINNED
)

type Predictor struct {
	handle     C.PredictorHandle
	outputSize uint32
}

type Model struct {
	Symbol []byte // json
	Params []byte // network
}

type Device struct {
	Type int
	Id   int
}

type InputNode struct {
	Key   string
	Shape []uint32
}

func NewPredictor(model Model, dev Device, input []InputNode) (*Predictor, error) {
	shapeInd := []uint32{0}
	shapeData := []uint32{}

	var b *C.char
	keys := C.malloc(C.size_t(len(input)) * C.size_t(unsafe.Sizeof(b)))
	defer C.free(unsafe.Pointer(keys))

	for i := 0; i < len(input); i++ {
		element := (**C.char)(unsafe.Pointer(uintptr(keys) + uintptr(i)*unsafe.Sizeof(b)))
		*element = C.CString(input[i].Key)
		shapeInd = append(shapeInd, uint32(len(input[i].Shape)))
		shapeData = append(shapeData, input[i].Shape...)
	}

	var handle C.PredictorHandle
	n, err := C.MXPredCreate((*C.char)(unsafe.Pointer(&model.Symbol[0])), (*C.char)(unsafe.Pointer(&model.Params[0])), C.size_t(len(model.Params)), C.int(dev.Type), C.int(dev.Id), C.mx_uint(len(input)), (**C.char)(keys), (*C.mx_uint)(&shapeInd[0]), (*C.mx_uint)(&shapeData[0]), &handle)

	for i := 0; i < len(input); i++ {
		element := (**C.char)(unsafe.Pointer(uintptr(keys) + uintptr(i)*unsafe.Sizeof(b)))
		C.free(unsafe.Pointer(*element))
	}

	if err != nil {
		return nil, err
	} else if n < 0 {
		return nil, GetLastError()
	}

	return &Predictor{handle, 0}, nil
}

func (p *Predictor) Free() {
	if p.handle != nil {
		C.MXPredFree(p.handle)
		p.handle = nil
	}
}

func (p *Predictor) Forward(key string, data []float32) error {
	if data != nil {
		k := C.CString(key)
		defer C.free(unsafe.Pointer(k))
		if n, err := C.MXPredSetInput(p.handle, k, (*C.mx_float)(&data[0]), C.mx_uint(len(data))); err != nil {
			return err
		} else if n < 0 {
			return GetLastError()
		}
	}

	if n, err := C.MXPredForward(p.handle); err != nil {
		return err
	} else if n < 0 {
		return GetLastError()
	}
	return nil
}

func (p *Predictor) GetOutput(index uint32) ([]float32, error) {
	if p.outputSize == 0 {
		var shapeData *C.mx_uint
		var shapeDim C.mx_uint
		if n, err := C.MXPredGetOutputShape(p.handle, C.mx_uint(index), (**C.mx_uint)(&shapeData), (*C.mx_uint)(&shapeDim)); err != nil {
			return nil, err
		} else if n < 0 {
			return nil, GetLastError()
		}

		var size uint32 = 1
		for i := 0; i < int(shapeDim); i++ {
			n := *(*C.mx_uint)(unsafe.Pointer(uintptr(unsafe.Pointer(shapeData)) + uintptr(i)*unsafe.Sizeof(size)))
			size *= uint32(n)
		}

		p.outputSize = size
	}

	size := p.outputSize
	data := make([]C.mx_float, size)
	if n, err := C.MXPredGetOutput(p.handle, C.mx_uint(index), (*C.mx_float)(&data[0]), C.mx_uint(size)); err != nil {
		return nil, err
	} else if n < 0 {
		return nil, GetLastError()
	}
	out := make([]float32, size)
	for i := 0; i < int(size); i++ {
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
