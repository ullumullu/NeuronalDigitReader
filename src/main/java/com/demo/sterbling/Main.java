/**
 * 
 */
package com.demo.sterbling;

import com.demo.sterbling.mnist.MNist;
import com.demo.sterbling.neuronal.NeuronalDigitNetwork;

/**
 * @author Sterbling
 *
 */
public class Main {

	/**
	 * @param args
	 * @throws InterruptedException 
	 */
	public static void main(String[] args) throws InterruptedException {

		final MNist trainData = new MNist("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz");
		final MNist testData = new MNist("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz");
		
		final NeuronalDigitNetwork ndn = new NeuronalDigitNetwork(new int[] {784, 30, 10});
		ndn.SGD(trainData.getTrainDataSet(), 30, 10, 3.0, testData.getTrainDataSet());
	}

}
