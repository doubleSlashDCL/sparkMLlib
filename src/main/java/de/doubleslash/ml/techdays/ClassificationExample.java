package de.doubleslash.ml.techdays;

import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class ClassificationExample {
	private static SparkSession spark;
	private static final double SQM_MUC = 73;
	private static final double PRICE_MUC = 1000;

	private static final double SQM_FN = 65;
	private static final double PRICE_FN = 700;

	public static void main(String[] args) {
		spark = initSparkSession();

		// Load training data
		Dataset<Row> training = spark.read().format("libsvm").load("data/mietkosten_fn_muc.txt");

		LinearSVCModel lsvcModel = getSVCModel(training);

		predictModel(lsvcModel, SQM_MUC, PRICE_MUC);
		predictModel(lsvcModel, SQM_FN, PRICE_FN);

		spark.stop();
	}

	private static SparkSession initSparkSession() {
		SparkSession spark = SparkSession.builder().appName("JavaLinearSVCExample").config("spark.master", "local")
				.getOrCreate();
		spark.sparkContext().setLogLevel("ERROR");
		return spark;
	}

	private static LinearSVCModel getSVCModel(Dataset<Row> training) {
		// Set parameter for Support Vector Machine
		LinearSVC lsvc = new LinearSVC().setMaxIter(10);
		// Fit the model
		LinearSVCModel lsvcModel = lsvc.fit(training);

		// Print the coefficients and intercept for LinearSVC
		System.out.println("Coefficients: " + lsvcModel.coefficients() + " Intercept: " + lsvcModel.intercept());
		return lsvcModel;
	}

	private static void predictModel(LinearSVCModel lsvcModel, double sqm, double price) {
		Vector testData = Vectors.dense(price, sqm);
		System.out.println("Prediction for " + sqm + "m² and " + price + "€: " + lsvcModel.predict(testData));
	}

}
