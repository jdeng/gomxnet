// https://github.com/gonum/floats/blob/master/floats.go
package predict

import (
	"sort"
)

// argsort is a helper that implements sort.Interface, as used by
// Argsort.
type argsort struct {
	s    []float32
	inds []int
}

func (a argsort) Len() int {
	return len(a.s)
}

func (a argsort) Less(i, j int) bool {
	return a.s[i] > a.s[j]
}

func (a argsort) Swap(i, j int) {
	a.s[i], a.s[j] = a.s[j], a.s[i]
	a.inds[i], a.inds[j] = a.inds[j], a.inds[i]
}

// Argsort sorts the elements of s while tracking their original order.
// At the conclusion of Argsort, s will contain the original elements of s
// but sorted in increasing order, and inds will contain the original position
// of the elements in the slice such that dst[i] = origDst[inds[i]].
// It panics if the lengths of dst and inds do not match.
func Argsort(dst []float32, inds []int) {
	if len(dst) != len(inds) {
		panic("floats: length of inds does not match length of slice")
	}
	for i := range dst {
		inds[i] = i
	}

	a := argsort{s: dst, inds: inds}
	sort.Sort(a)
}
