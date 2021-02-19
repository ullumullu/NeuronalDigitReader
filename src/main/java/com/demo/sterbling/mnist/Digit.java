/**
 * 
 */
package com.demo.sterbling.mnist;

/**
 * @author Sterbling
 *
 */
public class Digit {

	final int width;
	final int height;
	final int[] rawPicture;
	
	int correctAnswer;
	
	public Digit(byte[] rawData, int width, int height) {
		final int[] transformedData = new int[rawData.length];
		for(int indx = 0; indx < rawData.length; indx++) {
			transformedData[indx] = Byte.toUnsignedInt(rawData[indx]);
		}
		this.rawPicture = transformedData;
		this.width = width;
		this.height = height;
	}

	public int[] getRawPicture() {
		return rawPicture;
	}
	
	public double[] getScalarPicture() {
		final double[] result = new double[rawPicture.length];
		int position = 0;
		for(int pixel : rawPicture) {			
			result[position] = (pixel >= 30) ? 1.0 : 0.0;
			position++;
		}
		return result;
	}
	
	public int getWidth() {
		return width;
	}
	
	public int getHeight() {
		return height;
	}
	
	public void setCorrectAnswer(Integer correctAnswer) {
		this.correctAnswer = correctAnswer;
	}

	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder();
		int counter = 0;
		for(int pixl : rawPicture) {
			sb.append(String.format("%-3d ", pixl));
			counter++;
			if(counter % width == 0) {
				sb.append("\n");
			}
		}
		sb.append("Expected Number: ").append(correctAnswer).append("\n");
		return sb.toString();
	}

}
