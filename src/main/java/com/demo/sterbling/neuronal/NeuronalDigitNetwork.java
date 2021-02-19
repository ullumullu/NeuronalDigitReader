package com.demo.sterbling.neuronal;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

import org.jscience.mathematics.number.Float64;
import org.jscience.mathematics.vector.Float64Matrix;
import org.jscience.mathematics.vector.Float64Vector;

import com.demo.sterbling.mnist.TrainingDataTuple;

public class NeuronalDigitNetwork {

	final int numLayers;
	final int[] layerSizes;
	final List<Float64Vector> biases;
	final List<Float64Matrix> weights;

	public NeuronalDigitNetwork(final int[] layerSizes) {
		final Random rnd = new Random();
        this.numLayers = layerSizes.length;
        this.layerSizes = layerSizes;
        this.biases = initBiasList(() -> Float64.valueOf(rnd.nextGaussian()));        
        this.weights = initWeightList(() -> Float64.valueOf(rnd.nextGaussian()));
	}

	public Float64Vector feedforward(Float64Vector input) {		
		Float64Vector result = input;
		for(int indx = 0; indx < biases.size(); indx++) {
			final List<Float64> tempVector = new ArrayList<>();
			final Float64Vector biasLayer = biases.get(indx);
			final Float64Matrix weightMatrix = weights.get(indx);
			for(int indxNeuron = 0; indxNeuron < biasLayer.getDimension(); indxNeuron++) {
				Float64Vector weightLayer = weightMatrix.getRow(indxNeuron);
				final Float64 dotProduct = weightLayer.times(result);
				final Float64 biasSigmoid = biasLayer.get(indxNeuron);
				final Float64 z = dotProduct.plus(biasSigmoid);
				tempVector.add(sigmoid(z));
			}
			result = Float64Vector.valueOf(tempVector);
		}
		return result;
	}
	
	public void SGD(final List<TrainingDataTuple> trainingData, final int epochs, final int mini_batch_size, final double eta, final List<TrainingDataTuple> testData) {
		final int trainingDataSize = trainingData.size();
		for(int epochIndx = 0; epochIndx <  epochs; epochIndx++) {
			
			Collections.shuffle(trainingData);
			
			for(int miniBatchIndx = 0; miniBatchIndx < trainingDataSize; miniBatchIndx += mini_batch_size) {
				updateMiniBatch(trainingData.subList(miniBatchIndx, miniBatchIndx + mini_batch_size), eta);
			}
						
			if(testData != null) {
				System.out.printf("Epoch %d: %d / %d \n", epochIndx+1, evaluate(testData), testData.size());
			} else {
				System.out.printf("Epoch %d complete.\n", epochIndx+1);
			}
			
		}
	}
	
	private void updateMiniBatch(final List<TrainingDataTuple> batch, final double eta) {
		final List<Float64Vector> biasChangeRate = initBiasList(() -> Float64.ZERO);
		final List<Float64Matrix> weightChangerate = initWeightList(() -> Float64.ZERO);
		
		for(TrainingDataTuple tuple : batch) {
			final BackPropTuple backpropRes = backprop(tuple);
			final List<Float64Vector> biasDelta = backpropRes.getBackpropBiases(); 
			final List<Float64Matrix> weightDelta = backpropRes.getBackpropWeights();
			
			for(int indx = 0; indx < biasChangeRate.size(); indx++) {
				biasChangeRate.set(indx, biasChangeRate.get(indx).plus(biasDelta.get(indx)));
				weightChangerate.set(indx, weightChangerate.get(indx).plus(weightDelta.get(indx)));
			}
			
		}
		for(int indx = 0; indx < biasChangeRate.size(); indx++) {
			this.weights.set(indx, weights.get(indx).minus(weightChangerate.get(indx).times(Float64.valueOf(eta / batch.size()))));
			this.biases.set(indx, biases.get(indx).minus(biasChangeRate.get(indx).times(eta / batch.size())));
		}
		
	}
	
	private BackPropTuple backprop(final TrainingDataTuple tuple) {
		final List<Float64Vector> resultBackpropBias = initBiasList(() -> Float64.ZERO);
		final List<Float64Matrix> resultBackpropWeight = initWeightList(() -> Float64.ZERO);
		
		final List<Float64Vector> activations = new ArrayList<>();		
		final List<Float64Vector> zVectors = new ArrayList<>();
		
		Float64Vector activation = Float64Vector.valueOf(tuple.getDigit().getScalarPicture());
		activations.add(activation);
		for(int indx = 0; indx < biases.size(); indx++) {
			final Float64Vector bias = biases.get(indx);
			final Float64Matrix weight = weights.get(indx);
			final Float64Vector z = calculateLogisticFunction(weight, activation, bias);
			zVectors.add(z);
			activation = sigmoidVector(z);
			activations.add(activation);
		}

		Float64Vector delta = 
				multiplyElementwise(costDerivative(activations.get(activations.size()-1), tuple.getExpectedVector()), sigmoidPrimeVector(zVectors.get(zVectors.size()-1)));

//      nabla_b[-1] = delta
//      nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		resultBackpropBias.set(resultBackpropBias.size()-1, delta);
		resultBackpropWeight.set(resultBackpropWeight.size()-1, Float64Matrix.valueOf(delta).transpose().times(Float64Matrix.valueOf(activations.get(activations.size()-2))));
		
		
		for(int indx = 2; indx < this.numLayers; indx ++) {
			final Float64Vector zVector = zVectors.get(zVectors.size()-indx);
			final Float64Vector sp = sigmoidPrimeVector(zVector);
			delta = multiplyElementwise(this.weights.get(this.weights.size()-indx+1).transpose().times(delta), sp); // check transpose
			resultBackpropBias.set(resultBackpropBias.size()-indx, delta);
			resultBackpropWeight.set(resultBackpropWeight.size()-indx, Float64Matrix.valueOf(delta).transpose().times(Float64Matrix.valueOf(activations.get(activations.size()-indx-1))));
			
		}
//      # Note that the variable l in the loop below is used a little
//      # differently to the notation in Chapter 2 of the book.  Here,
//      # l = 1 means the last layer of neurons, l = 2 is the
//      # second-last layer, and so on.  It's a renumbering of the
//      # scheme in the book, used here to take advantage of the fact
//      # that Python can use negative indices in lists.
//      for l in xrange(2, self.num_layers):
//          z = zs[-l]
//          sp = sigmoid_prime(z)
//          delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
//          nabla_b[-l] = delta
//          nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
//      return (nabla_b, nabla_w)
		
		return new BackPropTuple(resultBackpropBias, resultBackpropWeight);
	}
	


		
	private Float64Vector costDerivative(final Float64Vector outputActivations, final Float64Vector expected) {
		return outputActivations.minus(expected);
	}
	
	private int evaluate(List<TrainingDataTuple> testData) {
		return testData.stream().mapToInt(tuple -> {		
			final Float64Vector resultVector = feedforward(Float64Vector.valueOf(tuple.getDigit().getScalarPicture()));
			int maxIndx = -1;
			double maxValue = 0.0;
			for(int indx = 0; indx < resultVector.getDimension(); indx++) {
				double tempValue = resultVector.getValue(indx);
				if(tempValue > maxValue) {
					maxValue = tempValue;
					maxIndx = indx;
				}
			}
			return (maxIndx == tuple.getExpected()) ? 1 : 0;
		}).sum();
		
	}

	private Float64 sigmoid(final Float64 z) {
		return Float64.valueOf(1.0 / (1.0 + Math.exp(-z.floatValue())));
	}

	private Float64 sigmoidPrime(final Float64 z) {
		return sigmoid(z).times((Float64.ONE.minus(sigmoid(z))));
	}
	
	private List<Float64Vector> initBiasList(Supplier<Float64> content) {
        final List<Float64Vector> result = new ArrayList<>();
		for(int indx = 1; indx < numLayers; indx++) {
        	final int neurons = layerSizes[indx];
        	final List<Float64> tempBiases = new ArrayList<>();
        	for(int layerIndx = 0; layerIndx < neurons; layerIndx++) {
        		tempBiases.add(content.get());
        	}
        	result.add(Float64Vector.valueOf(tempBiases));
        }
		return result;
	}
	
	private List<Float64Matrix> initWeightList(final Supplier<Float64> content) {
		final List<Float64Matrix> result = new ArrayList<>();
		for(int indx = 1; indx < numLayers; indx++) {
			final int neurons = layerSizes[indx];
			final int previousLayerNeurons = layerSizes[indx-1];
			final List<Float64Vector> weightsForNeuron = new ArrayList<>();
			for(int neuronIndx = 0; neuronIndx < neurons; neuronIndx++) {
				final List<Float64> tempPrevWeights = new ArrayList<>();
				for(int prevIndx = 0; prevIndx < previousLayerNeurons; prevIndx++) {
					tempPrevWeights.add(content.get());
				}
				weightsForNeuron.add(Float64Vector.valueOf(tempPrevWeights));
			}
			result.add(Float64Matrix.valueOf(weightsForNeuron));
		}
		return result;
	}
	
	// z = np.dot(w, activation)+b
	private Float64Vector calculateLogisticFunction(final Float64Matrix weight, final Float64Vector activation, final Float64Vector bias) {
		final List<Float64> tempVector = new ArrayList<>();
		for(int indxNeuron = 0; indxNeuron < bias.getDimension(); indxNeuron++) {
			Float64Vector weightLayer = weight.getRow(indxNeuron);
			final Float64 dotProduct = weightLayer.times(activation);
			final Float64 biasSigmoid = bias.get(indxNeuron);
			final Float64 z = dotProduct.plus(biasSigmoid);
			tempVector.add(z);
		}			
		return Float64Vector.valueOf(tempVector);
	}
	
	private Float64Vector sigmoidVector(final Float64Vector z) {
		final List<Float64> result = new ArrayList<>();
		for(int indx = 0; indx < z.getDimension(); indx++) {
			result.add(sigmoid(z.get(indx)));
		}
		return Float64Vector.valueOf(result);
	}
	
	private Float64Vector sigmoidPrimeVector(final Float64Vector z) {
		final List<Float64> result = new ArrayList<>();
		for(int indx = 0; indx < z.getDimension(); indx++) {
			result.add(sigmoidPrime(z.get(indx)));
		}
		return Float64Vector.valueOf(result);
	}
	
	private Float64Vector multiplyElementwise(final Float64Vector a, final Float64Vector b) {
		double[] result = new double[a.getDimension()];
		for(int indx = 0; indx < a.getDimension(); indx++) {
			result[indx] = a.getValue(indx) * b.getValue(indx);
		}
		return Float64Vector.valueOf(result);
	}
}
