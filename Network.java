import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.function.Divide;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Network {

	private double alpha = 0.007;

	private int MAX_EPOCS = 3700;
	
	private RealMatrix hiddenLayer;
	private RealMatrix hiddenMiddleLayer;
	private RealMatrix outputLayer;

	private RealVector hiddenLayerBias;
	private RealVector hiddenMiddleLayerBias;
	private RealVector outputLayerBias;

	private UnivariateFunction sigmoidal = new UnivariateFunction() {
		@Override
		public double value(double x) {
			return (1 / (1 + Math.pow(Math.E, -x)));
		}
	};

	private UnivariateFunction sigmoidalDerivative = new UnivariateFunction() {
		@Override
		public double value(double fnet) {
			return fnet * (1 - fnet);
		}
	};

	public Network(int inputSize,int hiddenLayerSize,int hiddenMiddleLayerSize, int outputLayerSize) {
		System.out.println("Alpha: "+alpha+" Epoc number: "+MAX_EPOCS+" hiddenLayerSize: "+hiddenLayerSize+" hiddenMiddleLayerSize: "+hiddenMiddleLayerSize+
				" outputLayerSize: "+ outputLayerSize);
		Random random = new Random();
		hiddenLayer = MatrixUtils.createRealMatrix(hiddenLayerSize, inputSize);
		hiddenMiddleLayer = MatrixUtils.createRealMatrix(hiddenMiddleLayerSize, hiddenLayerSize);
		outputLayer = MatrixUtils.createRealMatrix(outputLayerSize, hiddenMiddleLayerSize);
		for (int i = 0; i < hiddenLayerSize; i++) {
			for (int j = 0; j < inputSize; j++) {
				hiddenLayer.addToEntry(i, j, random.nextDouble());
			}
		}
		
		for (int i = 0; i < hiddenMiddleLayerSize; i++) {
			for (int j = 0; j < hiddenLayerSize; j++) {
				hiddenMiddleLayer.addToEntry(i, j, random.nextDouble());
			}
		}


		for (int i = 0; i < outputLayerSize; i++) {
			for (int j = 0; j < hiddenMiddleLayerSize; j++) {
				outputLayer.addToEntry(i, j, random.nextDouble());
			}
		}

		double biasData1[] = new double[hiddenLayerSize];
		for (int i = 0; i < biasData1.length; i++) {
			biasData1[i] = random.nextDouble();
		}
		double biasData2[] = new double[hiddenMiddleLayerSize];

		for (int i = 0; i < biasData2.length; i++) {
			biasData2[i] = random.nextDouble();
		}
		double biasData3[] = new double[outputLayerSize];

		for (int i = 0; i < biasData3.length; i++) {
			biasData3[i] = random.nextDouble();
		}

		hiddenLayerBias = MatrixUtils.createRealVector(biasData1);
		hiddenMiddleLayerBias = MatrixUtils.createRealVector(biasData2);
		outputLayerBias = MatrixUtils.createRealVector(biasData3);
	}

	public Network(Network network) {
		this.hiddenLayer = network.hiddenLayer.copy();
		this.outputLayer = network.outputLayer.copy();
	}

	public Network trainNetwork(RealMatrix inputMatrix, RealMatrix labelMatrix) {
		Network trained = new Network(this);
		RealVector input;
		RealVector netY; // net input layer
		RealVector netY1; // net second layer
		RealVector netZ; // net third layer
		RealVector y; // signal output layer input
		RealVector y1; // signal second layer
		RealVector z; // signal third layer
		RealVector delta; // blad third layer
		RealVector ro; // blad first layer
		RealVector ro1; // blad second layer
		double err = 0;
		List<Double> errorHistory = new LinkedList<>();
		double epocs = 0;
		while (epocs < MAX_EPOCS) {
			RealVector errVector = MatrixUtils.createRealVector(new double[] { 0, 0, 0 });
			for (int i = 0; i < inputMatrix.getColumnDimension(); i++) {
				input = inputMatrix.getColumnVector(i);
				netY = this.hiddenLayer.operate(input).add(this.hiddenLayerBias);
				y = netY.map(sigmoidal);
				netY1 = this.hiddenMiddleLayer.operate(y).add(this.hiddenMiddleLayerBias);
				y1 = netY1.map(sigmoidal);
				netZ = this.outputLayer.operate(y1).add(this.outputLayerBias);
				z = netZ.map(sigmoidal);
				RealVector sub = labelMatrix.getColumnVector(i).subtract(z); // d-y
				errVector = errVector.add((labelMatrix.getColumnVector(i).subtract(z)).map(new UnivariateFunction() {

					@Override
					public double value(double x) {
						return x * x;
					}
				}));
				RealVector deriv = z.map(sigmoidalDerivative);
				delta = sub.ebeMultiply(deriv);
				RealVector errProp = this.outputLayer.transpose().operate(delta);
				deriv = y1.map(sigmoidalDerivative);
				ro1 = errProp.ebeMultiply(deriv);
				errProp = this.hiddenMiddleLayer.transpose().operate(ro1);
				
				deriv = y.map(sigmoidal);
				 ro = errProp.ebeMultiply(deriv);
				for (int j = 0; j < hiddenLayer.getRowDimension(); j++) {
					RealVector update = input.mapMultiply(ro.getEntry(j) * alpha);
					hiddenLayer.setRowVector(j, hiddenLayer.getRowVector(j).add(update));
					update = ro.mapMultiply(alpha);
					hiddenLayerBias = hiddenLayerBias.add(update);
				}
				
				for (int j = 0; j < hiddenMiddleLayer.getRowDimension(); j++) {
					RealVector update = y.mapMultiply(ro1.getEntry(j) * alpha);
					hiddenMiddleLayer.setRowVector(j, hiddenMiddleLayer.getRowVector(j).add(update));
					update = ro1.mapMultiply(alpha);
					hiddenMiddleLayerBias = hiddenMiddleLayerBias.add(update);
				}

				for (int j = 0; j < outputLayer.getRowDimension(); j++) {
					RealVector update = y1.mapMultiply(delta.getEntry(j) * alpha);
					outputLayer.setRowVector(j, outputLayer.getRowVector(j).add(update));
					update = delta.mapMultiply(alpha);
					outputLayerBias = outputLayerBias.add(update);
				}
			}
			err = errVector.getEntry(0) + errVector.getEntry(1) + errVector.getEntry(2);
				err *= 0.5;
			++epocs;
			if (epocs % 50 == 0) 
				
				System.out.println(String.format("ERROR = %f  epocs = %d", err,(int)epocs));
		}
		return trained;
	}

	public RealVector calcOut(RealVector input) {
		RealVector netY = this.hiddenLayer.operate(input).add(this.hiddenLayerBias);
		RealVector y = netY.map(sigmoidal);
		RealVector netY1 = this.hiddenMiddleLayer.operate(y).add(this.hiddenMiddleLayerBias);
		RealVector y1 = netY1.map(sigmoidal);
		RealVector netZ = this.outputLayer.operate(y1).add(this.outputLayerBias);
		RealVector z = netZ.map(sigmoidal);
		return z;
	}


	public static void main(String[] args) {
		Object[] data = NetworkUtils
				.prepareData("resources/wine.data");
		RealMatrix inputMatrix = (RealMatrix) data[0];
		RealMatrix labelMatrix = (RealMatrix) data[1];
		RealMatrix inputMatrixTest = (RealMatrix) data[2];
		RealMatrix labelMatrixTest = (RealMatrix) data[3];
		
		
		Network network = new Network(13,6,4,3);
		
		
//		System.out.println(network.hiddenLayer);
//		System.out.println(network.calcOut(inputMatrix.getColumnVector(0)));
		network.trainNetwork(inputMatrix, labelMatrix);
//		System.out.println(network.hiddenLayer);
//		System.out.println(labelMatrix.getColumnVector(0));
//		System.out.println(network.calcOut(inputMatrix.getColumnVector(0)));
		int count = 0;
		for(int i = 0 ; i < inputMatrix.getColumnDimension(); i ++) {
			RealVector out = network.calcOut(inputMatrix.getColumnVector(i));
			out = out.map(new UnivariateFunction() {
				
				@Override
				public double value(double x) {
					// TODO Auto-generated method stub
					return Math.round(x);
				}
			});
			//System.out.println(out);
			//System.out.println(labelMatrixTest.getColumnVector(i));
			boolean rowne = out.equals(labelMatrix.getColumnVector(i));
//			System.out.println(out.equals(labelMatrix.getColumnVector(i)));
			if(rowne) count++;
		}
		double acc = new Divide().value(count, inputMatrix.getColumnDimension());
		System.out.println("Train Set Accuracy: "+acc);
		count = 0;
		for(int i = 0 ; i < inputMatrixTest.getColumnDimension(); i ++) {
			RealVector out = network.calcOut(inputMatrixTest.getColumnVector(i));
			out = out.map(new UnivariateFunction() {
				
				@Override
				public double value(double x) {
					// TODO Auto-generated method stub
					return Math.round(x);
				}
			});
			//System.out.println(out);
			//System.out.println(labelMatrixTest.getColumnVector(i));
			boolean rowne = out.equals(labelMatrixTest.getColumnVector(i));
//			System.out.println(out.equals(labelMatrix.getColumnVector(i)));
			if(rowne) count++;
		}
		acc = new Divide().value(count, inputMatrixTest.getColumnDimension());
		System.out.println("Test Set Accuracy: "+acc);
	}

}
