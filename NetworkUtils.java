import java.util.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class NetworkUtils {

	public static Object[] prepareData(String path) {
		RealMatrix matrix = null;
		RealMatrix matrix2 = null;
		RealMatrix inputMatrix = null;
		RealMatrix inputMatrixTest = null;
		RealMatrix labelsMatrix = null;
		RealMatrix labelsMatrixTest = null;

		try {
			List<String> lines = Files.readAllLines(Paths.get(path));
			final List<List<Double>> values = new LinkedList<>();
			for (int i = 0; i < lines.size(); i++) {
				String[] numbers = lines.get(i).split(",");
				for (int j = 0; j < numbers.length; j++) {
					if (j == 0) {
						values.add(new LinkedList<>());
					}
					values.get(i).add(Double.parseDouble(numbers[j]));
				}
			}
			double min = Double.MAX_VALUE;
			double max = Double.MIN_VALUE;
			double[][] data = new double[values.size()-44][];
			double[][] data2 = new double[values.size()-134][];
			int calc=1;
			int ii=0;
			int ii2=0;
			for (int i = 0; i < values.size(); i++) {
				if(calc==3) {
				data2[ii] = new double[values.get(i).size()];
				for (int j = 0; j < values.get(i).size(); j++) {
					data2[ii][j] = values.get(i).get(j);
					if (data2[ii][j] > max) {
						max = data2[ii][j];
					}
					if (data2[ii][j] < min) {
						min = data2[ii][j];
					}
				}
				calc=0;
				ii++;
			}else {
				data[ii2] = new double[values.get(i).size()];
				for (int j = 0; j < values.get(i).size(); j++) {
					data[ii2][j] = values.get(i).get(j);
					if (data[ii2][j] > max) {
						max = data[ii2][j];
					}
					if (data[ii2][j] < min) {
						min = data[ii2][j];
					}
				}
				calc++;
				ii2++;
			}
			}
			double maxminsub = max - min;
			for (int i = 0; i < data.length; i++) {
				for (int j = 1; j < data[i].length; j++) {
					data[i][j] = (data[i][j] - min) / maxminsub;
				}
			}
			for (int i = 0; i < data2.length; i++) {
				for (int j = 1; j < data2[i].length; j++) {
					data2[i][j] = (data2[i][j] - min) / maxminsub;
				}
			}
			matrix = MatrixUtils.createRealMatrix(data);
			Set<Integer> indexes = new LinkedHashSet<>();
			Random random = new Random();
			while (indexes.size() < matrix.getRowDimension()) {
				indexes.add(random.nextInt(matrix.getRowDimension()));
			}
			Iterator<Integer> it = indexes.iterator();
			RealMatrix copy = matrix.copy();
			for (int i = 0; i < copy.getRowDimension(); i++) {
				copy.setRowVector(it.next(), matrix.getRowVector(i));
			}
			matrix = copy;
			inputMatrix = matrix.getSubMatrix(0, matrix.getRowDimension() - 1, 1, matrix.getColumnDimension() - 1);
			inputMatrix = inputMatrix.transpose();
			labelsMatrix = matrix.getSubMatrix(0, matrix.getRowDimension() - 1, 0, 0);
			labelsMatrix = labelsMatrix.transpose();
			RealMatrix tmpMatrx = MatrixUtils.createRealMatrix(3, labelsMatrix.getColumnDimension());
			for (int i = 0; i < labelsMatrix.getColumnDimension(); i++) {
				double[] newData = { 0, 0, 0 };
				double v = labelsMatrix.getEntry(0, i);
				newData[(int) v - 1] = 1;
				tmpMatrx.setColumn(i, newData);
			}
			labelsMatrix = tmpMatrx;
			
			matrix2 = MatrixUtils.createRealMatrix(data2);
			inputMatrixTest = matrix2.getSubMatrix(0, matrix2.getRowDimension() - 1, 1, matrix2.getColumnDimension() - 1);
			inputMatrixTest = inputMatrixTest.transpose();
			labelsMatrixTest = matrix2.getSubMatrix(0, matrix2.getRowDimension() - 1, 0, 0);
			labelsMatrixTest = labelsMatrixTest.transpose();
			RealMatrix tmpMatrx2 = MatrixUtils.createRealMatrix(3, labelsMatrixTest.getColumnDimension());
			for (int i = 0; i < labelsMatrixTest.getColumnDimension(); i++) {
				double[] newData = { 0, 0, 0 };
				double v = labelsMatrixTest.getEntry(0, i);
				newData[(int) v - 1] = 1;
				tmpMatrx2.setColumn(i, newData);
				//System.out.println(labelsMatrixTest.getEntry(0, i)+" "+newData.);
			}
			labelsMatrixTest = tmpMatrx2;

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return new Object[] { inputMatrix, labelsMatrix,inputMatrixTest, labelsMatrixTest };
	}

}
