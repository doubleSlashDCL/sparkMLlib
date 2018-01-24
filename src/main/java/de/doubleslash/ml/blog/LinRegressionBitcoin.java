package de.doubleslash.ml.blog;

import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class LinRegressionBitcoin {
	public static void main(String[] args) {
		SparkSession spark = SparkSession.builder().appName("LinRegressionBitcoin").config("spark.master", "local")
				.getOrCreate();
		spark.sparkContext().setLogLevel("ERROR");

		// Load DataFrame from libsvm
		Dataset<Row> training = spark.read().format("libsvm").load("data/BitcoinKurs30.txt");
		// show content of the DataFrame
		training.show(false);

		// Set parameters to regression
		LinearRegression lr = new LinearRegression().setMaxIter(10);

		// Fit the model
		LinearRegressionModel lrModel = lr.fit(training);

		// Transform the model and test the quality of the result
		LinearRegressionTrainingSummary trainingSummary = lrModel.summary();

		System.out.println("Coefficients: " + lrModel.coefficients() + " Intercept: " + lrModel.intercept());
		System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
		System.out.println("r2: " + trainingSummary.r2());

		Vector nextDay = Vectors.dense(43154);
		System.out.println("Prediction: " + lrModel.predict(nextDay));
		spark.stop();
	}
}
