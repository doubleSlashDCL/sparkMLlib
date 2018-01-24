package de.doubleslash.ml.techdays;

import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class RegressionExample {
	private static SparkSession spark;
	private static final double SQM = 73;
	private static final double ROOMS = 3;

	public static void main(String[] args) {
		initSparkSession();

		// Load training data.
		Dataset<Row> trainingData = spark.read().format("libsvm").load("data/mietkosten_fn.txt");

		LinearRegressionModel lrModel = getTrainedModel(spark, trainingData);

		printModelData(trainingData, lrModel);

		predictValues(lrModel, SQM, ROOMS);

		spark.stop();
	}

	private static void initSparkSession() {
		spark = SparkSession.builder().appName("Java Spark SQL basic example").config("spark.master", "local")
				.getOrCreate();
		spark.sparkContext().setLogLevel("ERROR");
	}

	private static LinearRegressionModel getTrainedModel(SparkSession spark, Dataset<Row> trainingData) {
		// Setup the model
		LinearRegression lr = new LinearRegression().setMaxIter(10);

		// Fit the model.
		LinearRegressionModel lrModel = lr.fit(trainingData);
		return lrModel;
	}

	private static void printModelData(Dataset<Row> trainingData, LinearRegressionModel lrModel) {
		// Print input data
		System.out.println("Training data:");
		trainingData.show(false);

		// Print the coefficients and intercept for linear regression.
		System.out.println("Intercept: " + lrModel.intercept());
		System.out.println("Coefficients: " + lrModel.coefficients());

		// Summarize the model over the training set and print out some metrics.
		LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
		System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
		System.out.println("r2: " + trainingSummary.r2());
	}

	private static void predictValues(LinearRegressionModel lrModel, double sqm, double rooms) {
		Vector testData = Vectors.dense(sqm, rooms);
		System.out.println("Prediction for " + sqm + "m² and " + rooms + " rooms: " + lrModel.predict(testData));
	}
}
