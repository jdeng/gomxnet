package gomxnet

import (
	"image"
)

func InputFrom(m image.Image, rMean, gMean, bMean float32) ([]float32, error) {
	bounds := m.Bounds()
	height := bounds.Max.Y - bounds.Min.Y
	width := bounds.Max.X - bounds.Min.X
	out := make([]float32, height*width*3)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := m.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			out[y*width+x] = float32(b>>8) - bMean
			out[width*height+y*width+x] = float32(g>>8) - gMean
			out[2*width*height+y*width+x] = float32(r>>8) - rMean
		}
	}
	return out, nil
}
