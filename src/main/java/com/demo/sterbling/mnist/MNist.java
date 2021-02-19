/**
 * 
 */
package com.demo.sterbling.mnist;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

import org.apache.commons.io.IOUtils;

/**
 * @author Sterbling
 *
 */
public class MNist {
	
	final List<TrainingDataTuple> trainingDataSet;
	
	public MNist(final String trainFileName, final String labelFileName) {
		this(createInputTuple(trainFileName, labelFileName));
	}
		
	public MNist(final TrainInputTuple training) {
		if(training.getData() != null && training.getData().exists() && training.data.canRead()) {
			this.trainingDataSet = transformData(training);
		} else {
			this.trainingDataSet = null;
		}
	}
	
	private List<TrainingDataTuple> transformData(final TrainInputTuple trainingTuple) {
		final File dataFile = trainingTuple.getData();
		final File labelFile = trainingTuple.getLabels();
		
		try(final GZIPInputStream dataIs = new GZIPInputStream(new FileInputStream(dataFile));
			final GZIPInputStream labelIs = new GZIPInputStream(new FileInputStream(labelFile))) {
						
			System.out.println("Start transforming " + labelFile.getName());
			final int magicNumberLabel = readInt(labelIs);
			final int labelSize = readInt(labelIs);
			System.out.printf("Magic Number: %08x \n", magicNumberLabel);
			System.out.println("Number Labels: " + labelSize);
			
			final List<Integer> labels = new ArrayList<>(labelSize);
			final InputStream bufferedLabels = IOUtils.toBufferedInputStream(labelIs, labelSize);			
			int readLabels;
			while((readLabels = bufferedLabels.read()) != -1) {
				labels.add(readLabels);
			}
			
			System.out.println("Start transforming " + dataFile.getName());
			final int magicNumberData = readInt(dataIs);
			final int numberImgs = readInt(dataIs);
			final int rowSize = readInt(dataIs);
			final int colSize = readInt(dataIs);
			final int pictureSize = rowSize * colSize;
			
			System.out.printf("Magic Number: %08x \n", magicNumberData);
			System.out.println("Number Images: " + numberImgs);
			System.out.println("Row Size: " + rowSize);
			System.out.println("Col Size: "+ colSize);
			System.out.println("Total Size: " + numberImgs*rowSize*colSize);
			
			final List<TrainingDataTuple> trainingData = new ArrayList<>(numberImgs * pictureSize);
			
			final byte[] picture = new byte[pictureSize];
			final InputStream bufferedData = IOUtils.toBufferedInputStream(dataIs, numberImgs * pictureSize);			
			int read;
			int indx = 0;
			while((read = bufferedData.read(picture)) != -1) {
				final Digit newDigit = new Digit(picture, rowSize, colSize);
				trainingData.add(new TrainingDataTuple(newDigit, labels.get(indx++)));
				if(read != 784) {
					System.err.println("Read only " + read + " bytes");
				}
			}
						
			return trainingData;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	private int readInt(GZIPInputStream dataIs) throws IOException {
		final byte[] intBuffer = new byte[4];
		dataIs.read(intBuffer);
		ByteBuffer wrappedBuffer = ByteBuffer.wrap(intBuffer);
		return wrappedBuffer.getInt();
	}

	public List<TrainingDataTuple> getTrainDataSet() {
		return trainingDataSet;
	}
	
	
	private static TrainInputTuple createInputTuple(final String trainFileName, final String labelFileName) {
		//Get file from resources folder
		ClassLoader classLoader = MNist.class.getClassLoader();
		
		final File traindata = new File(classLoader.getResource(trainFileName).getFile());
		final File trainlabels = new File(classLoader.getResource(labelFileName).getFile());
		final TrainInputTuple train = new TrainInputTuple(traindata, trainlabels);
		return train;
	}
}
