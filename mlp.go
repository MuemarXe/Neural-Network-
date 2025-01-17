package main

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

//The neural Network
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

/*For example, the Gonum Product function allows us to perform the dot product operation on
two matrices, and I created a helper function that finds out the size of the matrix,
creates it and perform the operation before returning the resultant matrix*/

/*
The Dim function in the Go programming language is used to

	find the maximum between the difference of two provided arguments and 0.
*/
func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

// the apply function allows us to apply a function to the matrix
func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

// the scale function allows us to use a matrix i.e multiply a matrix by a scalar
func scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

// The multiply function multiplies two functions together--This is differnt from dot Product
func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

// The add and subtract functions allow to add or subtract a function to/from another
func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}
func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

// The addScalar function allows us to add a scalar value to each element in the matrix

// func addScalar(i float64, m mat.Matrix) mat.Matrix {
// 	r, c := m.Dims()
// 	a := make([]float64, r*c)
// 	for x := 0; x < r*c; x++ {
// 		a[x] = i
// 	}
// 	n := mat.NewDense(r, c, a)
// 	return add(m, n)
// }

//Neural network and Matrices
// Initializing the weights with a random set of numbers is one of the more important parameters
//For this we're going to use a function "randomArray" to create this random array of float64

func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}
	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
	/*The function uses the distuv package to create a uniformly distributed set of values
	  between the range of -1/sqrt(v) and 1/sqrt(v) where v is the size of the from layer */

}

// func addBiasNodeTo(m mat.Matrix, b float64) mat.Matrix {
// 	r, _ := m.Dims()
// 	a := mat.NewDense(r+1, 1, nil)

// 	a.Set(0, 0, b)
// 	for i := 0; i < r; i++ {
// 		a.Set(i+1, 0, m.At(i, 0))
// 	}
// 	return a
// }

// // pretty print a Gonum matrix
// func matrixPrint(X mat.Matrix) {
// 	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
// 	fmt.Printf("%v\n", fa)
// }

// Now that we have our neural network , the two main functions we can ask it to do is
//Either train itself with a set of data or predict values given a set of test data

/*From our hard work earlier on, we know that prediction means forward propagation through the network
while training means forward propagation first,
then back propagation later on to change the weights using some training data.
*/

// Since both train and prediction requires foward propagation, lets start with that first***

//WE DEFINE A FUNCTION 'PREDICT' TO PREDICT THE VALUES USING TRAINED NEURAL NETWORK

func (net Network) Predict(inputData []float64) mat.Matrix {
	//foward Propagation
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)
	return finalOutputs
	/*We start off with the inputs first, by creating a matrix called inputs to represent the input values
	. Next we find the inputs to hidden layer by applying the dot product between the hidden weights and the inputs,
	 creating a matrix called hiddenInputs:*/
}

// We apply our activation function, "sigmoid"on the hidden inputs to produce hiddenOutputs
func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
	//We repeat these two actions for final inputs and final outputs to produce finalInputs and finalOutputs
	//respectively and the prediction is the final outputs
}

// Let’s see how we do forward and back propagation in training.
func (net *Network) Train(inputData []float64, targetData []float64) {
	// feedfoward
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	// find errors
	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := subtract(targets, finalOutputs)
	hiddenErrors := dot(net.outputWeights.T(), outputErrors)

	// backpropagate
	net.outputWeights = add(net.outputWeights,
		scale(net.learningRate,
			dot(multiply(outputErrors, sigmoidPrime(finalOutputs)),
				hiddenOutputs.T()))).(*mat.Dense)
	net.hiddenWeights = add(net.hiddenWeights,
		scale(net.learningRate,
			dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),
				inputs.T()))).(*mat.Dense)
	/*The first thing we need to do after getting the final outputs is to determine the output errors.
	  This is relatively simple, we simply subtract our target data from the final outputs to get outputErrors:
	*/
	/* We use back propagation to calculate the hidden errors by applying the dot product
	   on the transpose of the output weights and the output errors.
	    This will give us hiddenErrors.*/
}

/*
Remember that we are subtracting this number from the weights. Since this is a negative number,

	we end up adding this to the weights, which is what we did.

To simplify the calculations we use a sigmoidPrime function, which is nothing more than doing sigP = sig(1 - sig)
*/
func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(m, subtract(ones, m))
	//Finally we do this twice to get the new hidden and output weights for our neural network.
}

// Saving the Results

func save(net Network) {
	h, err := os.Create("data/hweights.model")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.MarshalBinaryTo(h)
	}
	o, err := os.Create("data/oweights.model")
	defer o.Close()
	if err == nil {
		net.outputWeights.MarshalBinaryTo(o)
	}
}
func load(net *Network) {
	h, err := os.Open("data/hweights.model")
	defer h.Close()
	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open("data/oweights.model")
	defer o.Close()
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(o)
	}
	return
}

// Predicting individual files
func dataFromImage(filePath string) (pixels []float64) {
	// read the file
	imgFile, err := os.Open(filePath)
	defer imgFile.Close()
	if err != nil {
		fmt.Println("cannot read file:", err)

	}
	img, err := png.Decode(imgFile)
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}
	//create a grayscale image
	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {

			var rgba = img.At(x, y)
			gray.Set(x, y, rgba)
		}

	}
	pixels = make([]float64, len(gray.Pix))
	// populate the pixel array subtract Pix from 255 because
	//that's how the MNIST database was trained (in reveerse)
	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.999) + 0.001
	}
	return
	/*
		Each pixel in the image represents an value but we can’t use the normal RGBA,
		instead we need an image.Gray . From the image.Gray struct we get the Pix value and translate it
		into a float64 value instead.
		 The MNIST image is white on black, so we need to subtract each pixel value from 255.
	*/
}

/*
Once we have the pixel array, it’s quite straightforward.
We use a predictFromImage function that takes in the neural network and predicts
the digit from an image file. The results are an array of probabilities
where the index is the digit.What we need to do is to find the index and return it
*/
func predictFromImage(net Network, path string) int {
	input := dataFromImage(path)
	output := net.Predict(input)
	matrixPrint(output)
	best := 0
	highest := 0.0
	for i := 0; i < net.outputs; i++ {
		if output.At(i, 0) > highest {
			best = i
			highest = output.At(i, 0)
		}
	}
	return best
}
