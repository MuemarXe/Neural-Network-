package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"time"
	//"bytes"
	// "image"
	// "image/png"
	// "encoding/base64"
)

func main() {
	// 784 inputs -28x28 pixels, each pixel is an input
	//200 hidden neurons -an arbitrary number
	//10 outputs -digits 0-9
	//0.1 is the learning rate
	net := CreateNetwork(784, 200, 10, 0.1)

	mnist := flag.String("mnist", "", "Either train or predict to evaluate neural network")
	flag.Parse()
	// Train or Mass predict to determine effectiveness of the trained network
	switch *mnist {
	case "train":
		mnistTrain(&net)
		save(net)
	case "predict":
		load(&net)
		mnistPredict(&net)
	default:
		//do noting......
	}

}

/*
In the CSV format every line is an image,
and each column except the first represents a pixel. The first column is the label,

	which is the actual digit that the image is supposed to represent. In other words,
	 this is the target output.
	Since there are 28 x 28 pixels, this means there are 785 columns in every row.
*/
func mnistTrain(net *Network) {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()
	for epochs := 0; epochs < 5; epochs++ {
		testFile, _ := os.Open("mnist_dataset_train.csv")
		r := csv.NewReader(bufio.NewReader(testFile))
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			inputs := make([]float64, net.inputs)
			for i := range inputs {
				x, _ := strconv.ParseFloat(record[i], 64)
				inputs[i] = (x / 255.0 * 0.99) + 0.01
			}

			targets := make([]float64, 10)
			for i := range targets {
				targets[i] = 0.01
			}
			x, _ := strconv.Atoi(record[0])
			targets[x] = 0.99
			net.Train(inputs, targets)
		}
		testFile.Close()
	}
	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to train: %s\n", elapsed)
	/*
	   We open up the CSV file and read each record, then process each record.
	    For every record we read in we create an array that represents the inputs and an array that represents the targets.

	   For the inputs array we take each pixel from the record, and convert it to a value between 0.0 and 1.0 with 0.0
	    meaning a pixel with no value and 1.0 meaning a full pixel.

	   For the targets array, each element of the array represents the probability of the index being the target digit.
	    For example, if the target digit is 3, then the 4th element targets[3] would have a probability of 0.99 while the
	     rest would have a probability of 0.01.

	   Once we have the inputs and targets, we call the Train function of the network and pass it the inputs and targets.

	   You might notice that we ran this in ‘epochs’. Basically what we did was to run this multiple times because the
	   more times we run through the training the better trained the neural network will be. However if we over-train it,
	   the network will overfit, meaning it will adapt too well with the training data and will ultimately perform badly
	    with data that it hasn’t seen before.
	*/
}
func mnistPredict(net *Network) {
	t1 := time.Now()
	checkFile, _ := os.Open("mnist_dataset/mnist_test.csv")
	defer checkFile.Close()

	score := 0
	r := csv.NewReader(bufio.NewReader(checkFile))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]float64, net.inputs)
		for i := range inputs {
			if i == 0 {
				inputs[i] = 1.0
			}
			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = (x / 255.0 * 0.99) + 0.01
		}
		outputs := net.Predict(inputs)
		best := 0
		highest := 0.0
		for i := 0; i < net.outputs; i++ {
			if outputs.At(i, 0) > highest {
				best = i
				highest = outputs.At(i, 0)
			}
		}
		target, _ := strconv.Atoi(record[0])
		if best == target {
			score++
		}
	}

	elapsed := time.Since(t1)
	fmt.Printf("Time taken to check: %s\n", elapsed)
	fmt.Println("score:", score)
}
