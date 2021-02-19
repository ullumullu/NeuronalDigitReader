/**
 * 
 */
package com.demo.sterbling.neuronal;

import java.util.List;

import org.jscience.mathematics.vector.Float64Matrix;
import org.jscience.mathematics.vector.Float64Vector;

/**
 * @author Sterbling
 *
 */
public class BackPropTuple {
	
	final List<Float64Vector> backpropBiases;
	final List<Float64Matrix> backpropWeights;
	
	public BackPropTuple(List<Float64Vector> backpropBiases, List<Float64Matrix> backpropWeights) {
		super();
		this.backpropBiases = backpropBiases;
		this.backpropWeights = backpropWeights;
	}

	public List<Float64Vector> getBackpropBiases() {
		return backpropBiases;
	}

	public List<Float64Matrix> getBackpropWeights() {
		return backpropWeights;
	}
	
}
