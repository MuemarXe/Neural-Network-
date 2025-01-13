package main

import (
	"gomun.org/v1/gonum/mat"
)

// creating a simple 3 layer neural network(also known as a multilayer perceptron)

type Network struct {
	inputs        int
	hiddens       int
	outputs       int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
	learningRate  float64
	/*The fields inputs, hiddens and output define the number of
	neurons in each of the input, hidden and output layers (remember, this is a 3 layer network).
	 The hiddenWeights and outputWeights fields are matrices that represent the weights from the input
	 to hidden layers, and the hidden to output layers respectively*/
}

// a simple method to actually create the neural network
func CreateNetwork(input, hidden, output int, rate float64) (net Network) {
	net = Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		learningRate: rate,
	}
	net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, randomArray(net.inputs*net.hiddens, float64(net.inputs)))
	net.outputWeights = mat.NewDense(net.outputs, net.hiddens, randomArray(net.hiddens*net.outputs, float64(net.hiddens)))
	return
}
