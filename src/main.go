package main

import (
	"./gomxnet"
	"bufio"
	"fmt"
	"image"
	"io/ioutil"
	"os"

	"github.com/disintegration/imaging"
)

func main() {
	file := "cat15.jpg"
	reader, err := os.Open(file)
	if err != nil {
		panic(err)
	}

	img, _, _ := image.Decode(reader)
	img = imaging.Fill(img, 224, 224, imaging.Center, imaging.Lanczos)
	input, _ := gomxnet.InputFrom(img, 117.0, 117.0, 117.0)

	symbol, err := ioutil.ReadFile("../Inception-symbol.json")
	if err != nil {
		panic(err)
	}
	params, err := ioutil.ReadFile("../Inception-0009.params")
	if err != nil {
		panic(err)
	}

	pred, err := gomxnet.NewPredictor(symbol, params, gomxnet.CPU, 0, 1, []string{"data"}, []uint32{0, 4}, []uint32{1, 3, 224, 224})
	if err != nil {
		panic(err)
	}

	pred.Forward("data", input)
	output, _ := pred.GetOutput(0)

	index := make([]int, len(output))
	gomxnet.Argsort(output, index)

	synset, err := os.Open("../synset.txt")
	if err != nil {
		panic(err)
	}

	scanner := bufio.NewScanner(synset)

	dict := []string{}
	for scanner.Scan() {
		dict = append(dict, scanner.Text())
	}
	for i := 0; i < 50; i++ {
		fmt.Printf("%d: %f, %d, %s\n", i, output[i], index[i], dict[index[i]])
	}
}
