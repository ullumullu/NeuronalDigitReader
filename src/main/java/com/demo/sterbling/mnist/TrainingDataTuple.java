/**
 * 
 */
package com.demo.sterbling.mnist;

import java.util.Arrays;

import org.jscience.mathematics.vector.Float64Vector;

/**
 * 
 * 
 * @author Sterbling
 *
 */
public class TrainingDataTuple {

	final Digit digit;
	final int expected;
	
	public TrainingDataTuple(final Digit digit, final int expected) {
		this.digit = digit;
		this.expected = expected;
	}

	public Digit getDigit() {
		return digit;
	}

	public int getExpected() {
		return expected;
	}
	
	public Float64Vector getExpectedVector() {
		final double[] expectedVector = new double[10];
		Arrays.fill(expectedVector, 0);
		expectedVector[expected] = 1;
		return Float64Vector.valueOf(expectedVector);
	}

	
}
