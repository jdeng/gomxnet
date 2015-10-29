package main

import (
	"./gomxnet"
	"bufio"
	"flag"
	"fmt"
	"image"
	//	"image/jpeg"
	"io/ioutil"
	"os"

	"github.com/disintegration/imaging"
)

func main() {
	flag.Parse()
	args := flag.Args()
	file := "cat15.jpg"
	if len(args) >= 1 {
		file = args[0]
	}

	reader, err := os.Open(file)
	if err != nil {
		panic(err)
	}

	var batch uint32 = 2
	img, _, _ := image.Decode(reader)
	img = imaging.Fill(img, 224, 224, imaging.Center, imaging.Lanczos)

	/*
		test, _ := os.OpenFile("test.jpg", os.O_CREATE|os.O_WRONLY, 0644)
		jpeg.Encode(test, img, nil)
	*/

	symbol, err := ioutil.ReadFile("../Inception-symbol.json")
	if err != nil {
		panic(err)
	}
	params, err := ioutil.ReadFile("../Inception-0009.params")
	if err != nil {
		panic(err)
	}
	synset, err := os.Open("../synset.txt")
	if err != nil {
		panic(err)
	}

	pred, err := gomxnet.NewPredictor(gomxnet.Model{symbol, params}, gomxnet.Device{gomxnet.CPU, 0}, []gomxnet.InputNode{{"data", []uint32{batch, 3, 224, 224}}})
	if err != nil {
		panic(err)
	}

	input, _ := gomxnet.InputFrom([]image.Image{img, img}, gomxnet.ImageMean{117.0, 117.0, 117.0})
	pred.Forward("data", input)
	output, _ := pred.GetOutput(0)
	pred.Free()

	dict := []string{}
	scanner := bufio.NewScanner(synset)
	for scanner.Scan() {
		dict = append(dict, scanner.Text())
	}

	outputLen := uint32(len(output)) / batch
	var b uint32 = 0
	for ; b < batch; b++ {
		out := output[b*outputLen : (b+1)*outputLen]
		index := make([]int, len(out))
		gomxnet.Argsort(out, index)

		fmt.Printf("image #%d\n", b)
		for i := 0; i < 20; i++ {
			fmt.Printf("%d: %f, %d, %s\n", i, out[i], index[i], dict[index[i]])
		}
		fmt.Println("")
	}
}
