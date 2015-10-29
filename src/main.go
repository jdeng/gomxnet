package main

import (
	"image"
  	"os"
	"fmt"
	"bufio"
	"io/ioutil"
	"./gomxnet"

	"github.com/disintegration/imaging"
)

func main() {
	file := "cat15.jpg"
	reader, _ := os.Open(file)
	img, _, _ := image.Decode(reader)
	img = imaging.Fill(img, 224, 224, imaging.Center, imaging.Lanczos)
	input, _ := gomxnet.InputFrom(img, 117.0, 117.0, 117.0)

	symbol, _ := ioutil.ReadFile("../Inception-symbol.json")
	params, _ := ioutil.ReadFile("../Inception-0009.params")
	pred, _ := gomxnet.NewPredictor(symbol, params, gomxnet.CPU, 0, 1, []string{"data"}, []uint32{4}, []uint32{1, 3, 224, 224} )
	pred.Forward("data", input)
	output, _ := pred.GetOutput(0)

	index := make([]int, len(output))
	gomxnet.Argsort(output, index)

	synset, _ := os.Open("../synset.txt")
	scanner := bufio.NewScanner(synset)
	
	dict := []string{}
	for scanner.Scan() {
		dict = append(dict, scanner.Text())
	}
	for i:=0; i<5; i++ {
		fmt.Printf("%d: %d, %s\n", i, index[i], dict[index[i]])
	}
}

