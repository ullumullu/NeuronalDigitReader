/**
 * 
 */
package com.demo.sterbling.mnist;

import java.io.File;
import java.util.List;

/**
 * @author Sterbling
 *
 */
public class TrainInputTuple {

	
	final File data;
	final File labels;
	
	List<TrainingDataTuple> trainingSet;
	
	public TrainInputTuple(final File data, final File labels) {
		this.data = data;
		this.labels = labels;
	}

	public File getData() {
		return data;
	}

	public File getLabels() {
		return labels;
	}

	public List<TrainingDataTuple> getTrainingSet() {
		return trainingSet;
	}
	
	public void setTrainingSet(List<TrainingDataTuple> digits) {
		this.trainingSet = digits;
	}
	
}
