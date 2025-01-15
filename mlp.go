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

func (net*Neural)Train (inputData[]float64,targetData[]float64){
	// feedfoward
	inputs :=mat.NewDense (len(inputData),1,inputData)
	hiddenInputs := dot(net.hiddenWeights,inputs)
	hiddenOutputs := apply(sigmoid,hiddenOutputs)
	finalInputs := dot(net.outputWeights,hiddenOutputs)
	finalOutputs := apply(sigmoid,finalInputs)

	// find errors
	targets := mat.NewDense(len(targetData),1,targetData)
	outputErrors := subtract(targets,finalOutputs)
	hiddenErrors := dot(net.outputWeights.T(),outputErrors)

	// backpropagate
	net.outputWeights= add(net.outputWeights,
	                  scale(net.learningRate,
						  dot(multiply(outputErrors,sigmoidPrime(finalOutputs))),
						      hiddenOutputs.T())).(*mat.Dense)
	net.hidddenWeights = add (net.hiddenWeights,
	                  scale(net.learningRate,
						dot(multiply(hiddenErrors,sigmoidPrime(hiddenOutputs))
						  inputData.T()))).(*mat.Dense)

}
// Predict uses the neural network to predict the value given input data

func(net Network) Predict(inputData[]float64)mat.Matrix{
	//feedfoward
	inputs:= mat.NewDense(len(inputData),1,inputData)
	hiddenInputs := dot(net.hiddenWeights,inputs)
	hiddenOutputs := apply(sigmoid,hiddenOutputs)
	finalInputs := dot(net.outputWeights,hiddenOutputs)
	finalOutputs := apply(sigmoid,finalInputs)
	return finalOutputs

}
func