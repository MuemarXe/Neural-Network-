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
func sigmoid (r,c int ,z float64)float64{
	return 1.0/(1+math.Exp(-1*z))
}
func sigmoidPrime(m mat.Matrix)m.Matrix{
	rows,_ := m.Dims()
	o := make([]float64,rows)
	for i :=range o {
		o[i]=1
	}
	ones:= mat.NewDense(rows,1,o)
	return multiply (m,subtract(ones,m))
}
/*For example, the Gonum Product function allows us to perform the dot product operation on
 two matrices, and I created a helper function that finds out the size of the matrix, 
 creates it and perform the operation before returning the resultant matrix*/


 /*The Dim function in the Go programming language is used to
  find the maximum between the difference of two provided arguments and 0.
  */
func dot (m,n mat.Matrix)mat.Matrix{
	r,_ := m.Dims()
	_,c := n.Dims()
	o :=mat.NewDense(r,c,nil)
	o.Product(m,n)
	return o
}
//the apply function allows us to apply a function to the matrix
func apply (fn func(i,j int,v float64) float64,m mat.Matrix)mat.Matrix{
	r,c := m.Dims()
	o := mat.NewDense(r,c,nil)
	o.Apply(fn,m)
	return o
}
// the scale function allows us to use a matrix i.e multiply a matrix by a scalar
func scale (s float64,m mat.Matrix)mat.Matrix{
	r,c := m.Dims()
	o := mat.NewDense(r,c,nil)
	o.Scale(s,m)
	return o
}
// The multiply function multiplies two functions together--This is differnt from dot Product
func multiply(m,n mat.Matrix)mat.Matrix{
	r,c := m.Dims()
	o := mat.NewDense(r,c,nil)
	o.MulElem(m,n)
	return o
}
// The add and subtract functions allow to add or subtract a function to/from another
func add(m,n mat.Matrix)mat.Matrix{
	r,c := m.Dims()
	o := mat.Dense(r,c,nil)
	o.Add(m,n)
	return o
}
func subtract(m,n mat.Matrix)mat.Matrix{
	r,c := m.Dims()
	o := mat.NewDense(r,c,nil)
	o.Sub(m,n)
	return o
}
// The addScalar function allows us to add a scalar value to each element in the matrix

func addScalar (i float64,m mat.Matrix)mat.Matrix{
	r,c := m.Dims()
	a := make([]float64,r*c)
	for x := 0; x<r*c;x++{
		a[x]=i
	}
	n:= mat.NewDense(r,c,a)
	return add(m,n)
}