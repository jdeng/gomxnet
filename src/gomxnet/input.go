package gomxnet

import (
	"fmt"
	"image"
)

type ImageMean struct {
	R, G, B float32
}

func InputFrom(imgs []image.Image, mean ImageMean) ([]float32, error) {
	if len(imgs) == 0 {
		return nil, fmt.Errorf("No image")
	}
	height := imgs[0].Bounds().Max.Y - imgs[0].Bounds().Min.Y
	width := imgs[0].Bounds().Max.X - imgs[0].Bounds().Min.X

	out := make([]float32, height*width*3*len(imgs))
	for i := 0; i < len(imgs); i++ {
		m := imgs[i]
		bounds := m.Bounds()
		h := bounds.Max.Y - bounds.Min.Y
		w := bounds.Max.X - bounds.Min.X
		if h != height || w != width {
			return nil, fmt.Errorf("Size not matched")
		}
		start := width * height * 3 * i
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				r, g, b, _ := m.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
				out[start+y*width+x] = float32(b>>8) - mean.B
				out[start+width*height+y*width+x] = float32(g>>8) - mean.G
				out[start+2*width*height+y*width+x] = float32(r>>8) - mean.R
			}
		}

	}
	return out, nil
}
